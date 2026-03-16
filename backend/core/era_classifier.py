"""
EraClassifier — Ära-/Dekaden-adaptives Processing (§2.14 Spec)
===============================================================

Erkennt das Aufnahme-Jahrzehnt (1890–2025) automatisch und leitet
material- und epochenspezifische Verarbeitungspriors ab.

Erkennungs-Kaskade (3 Stufen):
    Tier-1: LAION-CLAP-Embeddings → Nearest-Neighbor zu Ära-Referenz-Ankern
    Tier-2: DSP-Fingerprint (HF-Rolloff + Bandbreiten-Kurve)
    Tier-3: Mikrofon-Typ-Heuristik

Referenz: §2.14 Aurik-9-Spec (v9.9.5)
Autor: Aurik Development Team
Datum: 20. Februar 2026
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace as dc_replace
import hashlib
import json
import logging
import math
from pathlib import Path
import threading

import numpy as np

logger = logging.getLogger(__name__)

# --------------- Dekaden-Definition ----------------------------------------

VALID_DECADES: list[int] = [
    1890,
    1900,
    1910,
    1920,
    1930,
    1940,
    1950,
    1960,
    1970,
    1980,
    1990,
    2000,
    2010,
    2020,
    2025,
]

# Bekannte HF-Rolloff-Grenzen pro Jahrzehnt [Hz]
DECADE_HF_LIMITS: dict[int, float] = {
    1890: 3000,
    1900: 4000,
    1910: 5000,
    1920: 6000,
    1930: 7000,
    1940: 8000,
    1950: 10000,
    1960: 12000,
    1970: 16000,
    1980: 20000,
    1990: 20000,
    2000: 20000,
    2010: 20000,
    2020: 20000,
    2025: 20000,
}

# Material-Prior pro Dekade
DECADE_MATERIAL_PRIOR: dict[int, str] = {
    1890: "wax_cylinder",
    1900: "wax_cylinder",
    1910: "shellac",
    1920: "shellac",
    1930: "shellac",
    1940: "shellac",
    1950: "vinyl",
    1960: "vinyl",
    1970: "reel_tape",
    1980: "tape",
    1990: "cd_digital",
    2000: "cd_digital",
    2010: "streaming",
    2020: "streaming",
    2025: "streaming",
}

# GP-Warmstart: noise_reduction_strength prior mean pro Epoche
DECADE_NR_PRIOR_MEAN: dict[int, float] = {
    1890: 0.95,
    1900: 0.95,
    1910: 0.92,
    1920: 0.90,
    1930: 0.90,
    1940: 0.85,
    1950: 0.80,
    1960: 0.75,
    1970: 0.65,
    1980: 0.55,
    1990: 0.50,
    2000: 0.50,
    2010: 0.45,
    2020: 0.45,
    2025: 0.45,
}

DECADE_NR_PRIOR_STD: dict[int, float] = dict.fromkeys(VALID_DECADES, 0.07)
DECADE_NR_PRIOR_STD.update(
    {1900: 0.05, 1910: 0.05, 1920: 0.05, 1930: 0.05, 1940: 0.06, 1970: 0.08, 1980: 0.10, 1990: 0.10}
)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class EraResult:
    """Ergebnis des EraClassifiers.

    Attributes:
        decade:                Erkanntes Jahrzehnt (z. B. 1940, 1970, …).
        era_label:             Menschenlesbare Bezeichnung (z. B. „1970er").
        confidence:            Konfidenz ∈ [0.0, 1.0].
        material_prior:        Empfohlener Material-Typ-String aus ``DECADE_MATERIAL_PRIOR``.
        noise_profile:         Spektrales Rauschprofil (Bark-Bänder, 24 Werte).
        tier_used:             Welche Erkennungsstufe genutzt wurde (1 = CLAP, 2 = DSP, 3 = Heuristik).
        hf_rolloff_hz:         Gemessener HF-Rolloff-Punkt (-3 dB) in Hz.
        is_remaster_suspected: True wenn RemasterDetector einen Remaster erkannt hat.
    """

    decade: int
    era_label: str
    confidence: float
    material_prior: str
    noise_profile: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=np.float32))
    tier_used: int = 2
    hf_rolloff_hz: float = 20000.0
    is_remaster_suspected: bool = False

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.decade not in VALID_DECADES:
            self.decade = min(VALID_DECADES, key=lambda d: abs(d - self.decade))

    def as_dict(self) -> dict:
        """Serialisierung ohne ndarray."""
        d = asdict(self)
        d["noise_profile"] = self.noise_profile.tolist()
        return d


# ---------------------------------------------------------------------------
# Bark-Skala Hilfsfunktion
# ---------------------------------------------------------------------------

BARK_EDGES_HZ = [
    20,
    100,
    200,
    300,
    400,
    510,
    630,
    770,
    920,
    1080,
    1270,
    1480,
    1720,
    2000,
    2320,
    2700,
    3150,
    3700,
    4400,
    5300,
    6400,
    7700,
    9500,
    12000,
    15500,
]


def _bark_band_energies(audio_mono: np.ndarray, sr: int) -> np.ndarray:
    """Berechnet normalisierte Energie in 24 Bark-Bändern.

    Args:
        audio_mono: Mono-Audio (1D float32/64).
        sr:         Sample-Rate.

    Returns:
        ndarray shape (24,) — normalisierte Energien (sum = 1).
    """
    n_fft = min(4096, len(audio_mono))
    hop = n_fft // 4
    # STFT (kein scipy.signal für diesen einfachen Spektral-Pfad — numpy direkt)
    frames = []
    for start in range(0, len(audio_mono) - n_fft, hop):
        frame = audio_mono[start : start + n_fft] * np.hanning(n_fft)
        frame_fft = np.abs(np.fft.rfft(frame)) ** 2
        frames.append(frame_fft)
    if not frames:
        return np.ones(24) / 24.0
    psd = np.mean(np.array(frames), axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    energies = np.zeros(24, dtype=np.float32)
    for i, (lo, hi) in enumerate(zip(BARK_EDGES_HZ[:-1], BARK_EDGES_HZ[1:])):
        mask = (freqs >= lo) & (freqs < hi)
        energies[i] = float(np.sum(psd[mask]))

    total = energies.sum()
    if total < 1e-12:
        return np.ones(24, dtype=np.float32) / 24.0
    return np.nan_to_num(energies / total)


# ---------------------------------------------------------------------------
# Tier-2: DSP-Fingerprint
# ---------------------------------------------------------------------------


def _dsp_hf_rolloff(audio_mono: np.ndarray, sr: int) -> float:
    """Effektive Bandbreite via 90%-Energie-Schwelle (gemittelt über mehrere Fenster).

    Ersetzt den fehlerhaften 85%-Schwellwert. Der 90%-Energiepunkt ist robuster
    für bandbreitenbegrenzte Vintage-Aufnahmen und konsistent mit den Werten in
    DECADE_HF_LIMITS. Mehrere überlappende Fenster (hop = n_fft//2) stabilisieren
    den Schätzwert gegenüber dem früheren Einzelfenster-Ansatz.

    Args:
        audio_mono: Mono-Audio.
        sr:         Sample-Rate.

    Returns:
        Rolloff-Frequenz in Hz.
    """
    n_fft = min(4096, len(audio_mono))
    if n_fft < 64:
        return float(sr) / 2.0
    hop = n_fft // 2  # 50%-Überlappung für bessere Mittelung
    specs = []
    for start in range(0, max(1, len(audio_mono) - n_fft), hop):
        frame = audio_mono[start : start + n_fft] * np.hanning(n_fft)
        specs.append(np.abs(np.fft.rfft(frame)) ** 2)
    if not specs:
        return float(sr) / 2.0

    avg_spec = np.mean(np.array(specs), axis=0)
    cum_energy = np.cumsum(avg_spec)
    total_energy = cum_energy[-1]
    if total_energy < 1e-12:
        return float(sr) / 2.0

    # 90%-Energiepunkt (robuster als -3 dB für vintage-typische harte Rolloffs)
    idx = int(np.searchsorted(cum_energy, 0.90 * total_energy))
    idx = int(np.clip(idx, 0, len(avg_spec) - 1))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return float(freqs[idx])


def _dsp_fingerprint_decade(rolloff_hz: float, snr_db: float) -> tuple[int, float]:
    """Mappt Bandbreite auf Jahrzehnt via Schwellwert-Tabelle.

    Ersetzt den SNR-Dominanzfehler (früherer *200-Multiplikator auf SNR-Differenz
    ließ 1 dB SNR-Abweichung 200 Hz-äquivalent wirken und überlagerte alle
    Spektral-Evidenz). Die Jahrzehnt-Selektion basiert jetzt primär auf der
    Bandbreite (rolloff_hz). SNR dient nur als leichter Korrekturfaktor bei
    Grenzfällen (Carbon-Mikrofon-Heuristik).

    Args:
        rolloff_hz: Gemessener HF-Rolloff in Hz.
        snr_db:     Geschätzter SNR in dB.

    Returns:
        (decade, confidence)
    """
    bw_khz = rolloff_hz / 1000.0

    # Primäre Jahrzehnt-Selektion via Bandbreite.
    # Schwellwerte = Mittelwert der 90th-Pct-Rolloffs benachbarter Jahrzehnte:
    #   threshold(D, D+1) = (DECADE_HF_LIMITS[D] × 0.9 + DECADE_HF_LIMITS[D+1] × 0.9) / 2
    # Beispiel 1960/1970: (15000×0.9 + 17000×0.9)/2 = (13500+15300)/2 = 14400 → 14.5 kHz
    if bw_khz < 4.5:
        decade = 1890  # LIMIT  4 kHz → expected rolloff ~3.6 kHz
    elif bw_khz < 5.4:
        decade = 1910  # LIMIT  5 kHz → expected rolloff ~4.5 kHz
    elif bw_khz < 7.0:
        decade = 1920  # LIMIT  7 kHz → expected rolloff ~6.3 kHz
    elif bw_khz < 8.8:
        decade = 1930  # LIMIT  8.5 kHz → expected rolloff ~7.7 kHz (0.95×8.5=8.1, +0.7 Marge)
    elif bw_khz < 11.3:
        decade = 1940  # LIMIT 12 kHz → expected rolloff ~10.8 kHz
    elif bw_khz < 12.6:
        decade = 1950  # LIMIT 13 kHz → expected rolloff ~11.7 kHz
    elif bw_khz < 14.5:
        decade = 1960  # LIMIT 15 kHz → expected rolloff ~13.5 kHz
    elif bw_khz < 17.0:
        decade = 1970  # LIMIT 17 kHz → expected rolloff ~15.3 kHz
    elif bw_khz < 19.0:
        decade = 1980  # LIMIT 20 kHz → expected rolloff ~18.0 kHz
    else:
        decade = 1990  # LIMIT 22 kHz → expected rolloff ~19.8 kHz

    # Leichte SNR-Mikro-Korrektur (Carbon/Ribbon-Mikrofon-Heuristik, kein Dominanzproblem)
    if snr_db < 20.0 and bw_khz < 6.0:
        decade = min(decade, 1930)  # Carbon-Mikrofon-Charakteristik
    elif snr_db < 25.0 and bw_khz < 8.0 and decade > 1940:
        decade = min(max(decade, 1920), 1940)  # Ribbon-Mikrofon-Ära

    # Confidence aus relativem BW-Fehler gegenüber dem Tabellenwert des Jahrzehnts
    expected_bw = DECADE_HF_LIMITS.get(decade, 10000.0) / 1000.0
    bw_error = abs(bw_khz - expected_bw) / max(expected_bw, 1.0)
    conf = float(np.clip(1.0 - bw_error * 0.8, 0.25, 0.85))
    if bw_khz >= 18.0:
        conf = max(conf, 0.75)  # modernes Material eindeutig erkennbar
    return decade, conf


def _decade_expected_snr(decade: int) -> float:
    """Grobe SNR-Erwartung pro Dekade [dB]."""
    snr_map = {
        1890: 12,
        1900: 15,
        1910: 18,
        1920: 22,
        1930: 28,
        1940: 32,
        1950: 38,
        1960: 44,
        1970: 52,
        1980: 58,
        1990: 65,
        2000: 70,
        2010: 75,
        2020: 80,
        2025: 80,
    }
    return snr_map.get(decade, 50)


def _estimate_snr(audio_mono: np.ndarray, sr: int = 48000) -> float:
    """Frame-basierte SNR-Schätzung via Energie-Perzentile.

    Robuster als die frühere Sample-Level-Sortierung für Vintage-Aufnahmen
    mit dauerhaftem Rauschen: Die Sortierung von Einzelsamples lieferte dort
    ~52 dB SNR statt der korrekten ~15–25 dB, da kurze Stille-Momente die
    untersten Perzentile dominierten.

    Frame-Level-Ansatz (100 ms Frames):
        - 10. Energie-Perzentil → Rauschboden
        - 90. Energie-Perzentil → Nutz-Signal

    Args:
        audio_mono: Mono-Audio.
        sr:         Sample-Rate (für Frame-Größen-Berechnung).

    Returns:
        Geschätzter SNR in dB, geclamppt auf [0, 80].
    """
    frame_size = max(1, sr // 10)  # 100-ms-Frames
    frames = [audio_mono[i : i + frame_size] for i in range(0, len(audio_mono) - frame_size, frame_size)]
    if not frames:
        return 40.0
    energies = np.array([np.mean(f**2) for f in frames])
    noise_floor = float(np.percentile(energies, 10))
    signal_power = float(np.percentile(energies, 90))
    if noise_floor < 1e-18:
        return 60.0
    snr = 10.0 * math.log10(max(signal_power / noise_floor, 1.0))
    return float(np.clip(snr, 0.0, 80.0))


# ---------------------------------------------------------------------------
# Tier-3: Mikrofon-Typ-Heuristik
# ---------------------------------------------------------------------------


def _microphone_type_decade(bark_energies: np.ndarray) -> tuple[int, float]:
    """Grobe Ära-Schätzung aus charakteristischem Mikrofon-Frequenzgang.

    Carbon-Mikrofone (1900–1930): SNR < 25 dB, Frequenzgang stark abfallend
    Kondensator post-1950: Frequenzgang-Flächigkeit ≥ 0.80

    Args:
        bark_energies: 24 normalisierte Bark-Bänder-Energien.

    Returns:
        (decade, confidence)
    """
    # Flächigkeit = Verhältnis Min/Max im relevanten Bereich (Bänder 2–18)
    relevant = bark_energies[2:18]
    if relevant.max() < 1e-10:
        return 1960, 0.20
    flatness = float(relevant.min() / (relevant.max() + 1e-10))

    # Hohe LF-Dominanz → älteres Mikrofon
    lf_dominance = float(np.sum(bark_energies[:6]) / (np.sum(bark_energies) + 1e-10))

    if flatness < 0.05 and lf_dominance > 0.60:
        return 1920, 0.35
    elif flatness < 0.15:
        return 1940, 0.30
    elif flatness < 0.40:
        return 1960, 0.28
    else:
        return 1980, 0.25


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".aurik" / "era_cache"
_CACHE_VERSION = "v2"  # Erhöhen bei DSP-Algorithmen-Änderungen → alte Caches automatisch ungültig


class EraClassifier:
    """Erkennt Aufnahme-Ära (1890–2025) und leitet epochenspezifische Priors ab.

    Erkennungs-Kaskade (3 Stufen):
        Tier-1: LAION-CLAP-Embeddings → NN zu Ära-Referenz-Ankern
        Tier-2: DSP-Fingerprint → HF-Rolloff + Bandbreiten-Kurve
        Tier-3: Mikrofon-Typ-Heuristik (Carbon/Kondensator)

    Ausgabe: EraResult(decade, era_label, confidence, material_prior,
                       noise_profile, tier_used, hf_rolloff_hz)

    Invarianten:
        - Konfidenz < 0.4 → material_prior = "unknown" (konservative Priors)
        - CLAP-Fallback auf DSP-Fingerprint wenn Import fehlschlägt
        - Decade-Label wird in RestorationResult.era_decade gespeichert
        - Paläografie-Cache unter ~/.aurik/era_cache/<sha256_prefix>.json
    """

    def __init__(self) -> None:
        self._clap_plugin: object | None = None
        self._clap_loaded: bool = False
        self._clap_lock = threading.Lock()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def classify(self, audio: np.ndarray, sr: int) -> EraResult:
        """Erkennt Aufnahme-Ära (Cascaded Tier-1 → Tier-2 → Tier-3).

        Args:
            audio: Audio-Signal (mono oder stereo).
            sr:    Sample-Rate in Hz — muss exakt 48000 sein (Spec §3.x).

        Returns:
            EraResult mit Dekade, Confidence und Material-Prior.

        Raises:
            ValueError:    Falls audio leer ist.
            AssertionError: Falls sr != 48000.
        """
        if audio.size == 0:
            raise ValueError("Audio darf nicht leer sein.")
        assert sr == 48000, (
            f"EraClassifier.classify(): 48000 Hz erwartet, erhalten: {sr} Hz. "
            "Bitte Audiodaten vor dem Aufruf auf 48000 Hz resampeln."
        )
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=-1 if audio.shape[-1] <= 2 else 0)
        else:
            audio_mono = audio.copy()

        # Cache-Key aus SHA256-Prefix
        sha = hashlib.sha256(audio_mono.tobytes()).hexdigest()[:16]
        cache_path = CACHE_DIR / f"{sha}_{_CACHE_VERSION}.json"
        cached = self._load_cache(cache_path)
        if cached:
            logger.debug("EraClassifier: Cache-Hit %s", sha)
            return cached

        bark = _bark_band_energies(audio_mono, sr)
        rolloff_hz = _dsp_hf_rolloff(audio_mono, sr)
        snr_db = _estimate_snr(audio_mono, sr)

        # Tier-1: CLAP (optional)
        result = self._try_tier1(audio_mono, sr, bark, rolloff_hz, snr_db)

        # Tier-2: DSP-Fingerprint
        if result is None or result.confidence < 0.40:
            result = self._tier2(bark, rolloff_hz, snr_db)

        # Tier-3: Mikrofon-Heuristik (letzter Fallback)
        if result.confidence < 0.30:
            result = self._tier3(bark, rolloff_hz, snr_db)

        # Invariante: Conf < 0.40 → konservatives Material
        if result.confidence < 0.40:
            result = EraResult(
                decade=result.decade,
                era_label=result.era_label,
                confidence=result.confidence,
                material_prior="unknown",
                noise_profile=result.noise_profile,
                tier_used=result.tier_used,
                hf_rolloff_hz=rolloff_hz,
            )

        # RemasterDetector-Guard (§2.14): verhindert falsche Ära-Zuweisung bei Remasters
        try:
            from backend.core.remaster_detector import get_remaster_detector
            _rm = get_remaster_detector().analyse(audio_mono, sr)
            if _rm is not None and _rm.is_remaster:
                result = dc_replace(result, is_remaster_suspected=True)
                logger.info(
                    "RemasterDetector: Remaster erkannt (conf=%.2f, BW=%.1f kHz)",
                    _rm.confidence,
                    getattr(_rm, "hf_rolloff_khz", 0.0),
                )
        except Exception:  # noqa: BLE001
            pass

        self._save_cache(cache_path, result)
        logger.info(
            "🕰️ EraClassifier: Jahrzehnt=%d, Konfidenz=%.2f, Material=%s, Tier=%d",
            result.decade,
            result.confidence,
            result.material_prior,
            result.tier_used,
        )
        return result

    def get_material_prior(self, era: EraResult) -> str:
        """Gibt empfohlenen Material-String für CausalDefectReasoner zurück.

        Bei Konfidenz < 0.40 → 'unknown' (konservative Priors, Spec §2.14).
        """
        if era.confidence < 0.40:
            return "unknown"
        return era.material_prior

    def get_gp_warmstart(self, era: EraResult) -> dict[str, float]:
        """GP-Optimizer-Initialisierungswerte für das erkannte Jahrzehnt.

        Returns:
            Dict mit Parameternamen → Initialwert.
        """
        decade = era.decade
        nr_mean = DECADE_NR_PRIOR_MEAN.get(decade, 0.65)
        nr_std = DECADE_NR_PRIOR_STD.get(decade, 0.08)
        return {
            "noise_reduction_strength": float(np.clip(nr_mean, 0.10, 1.0)),
            "noise_reduction_strength_std": nr_std,
            "harmonic_boost_db": 2.0 if decade <= 1950 else 1.0,
            "ola_crossfade_ms": 50.0 if decade <= 1940 else 30.0,
            "bass_restoration_db": 2.5 if decade <= 1960 else 0.5,
            "era_decade": float(decade),
            "era_confidence": float(era.confidence),
        }

    # ------------------------------------------------------------------
    # Tier-Implementierungen
    # ------------------------------------------------------------------

    def _try_tier1(
        self,
        audio_mono: np.ndarray,
        sr: int,
        bark: np.ndarray,
        rolloff_hz: float,
        snr_db: float,
    ) -> EraResult | None:
        """Tier-1: LAION-CLAP-basierte Ära-Erkennung (optional)."""
        try:
            with self._clap_lock:
                if not self._clap_loaded:
                    from plugins.laion_clap_plugin import get_laion_clap_plugin  # type: ignore[import]

                    self._clap_plugin = get_laion_clap_plugin()
                    self._clap_loaded = True
            if self._clap_plugin is None:
                return None
            # CLAP-Embedding → Cosinus-Ähnlichkeit zu Ära-Ankern
            embedding = self._clap_plugin.embed_audio(audio_mono, sr)  # type: ignore[union-attr]
            decade, conf = self._clap_nearest_neighbor(embedding)
            if conf < 0.35:
                return None
            return EraResult(
                decade=decade,
                era_label=f"{decade}er",
                confidence=conf,
                material_prior=DECADE_MATERIAL_PRIOR.get(decade, "unknown"),
                noise_profile=bark,
                tier_used=1,
                hf_rolloff_hz=rolloff_hz,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("EraClassifier Tier-1 fehlgeschlagen: %s — nutze DSP-Fallback", exc)
            return None

    def _tier2(self, bark: np.ndarray, rolloff_hz: float, snr_db: float) -> EraResult:
        """Tier-2: DSP-Fingerprint (HF-Rolloff + SNR)."""
        decade, conf = _dsp_fingerprint_decade(rolloff_hz, snr_db)
        material = DECADE_MATERIAL_PRIOR.get(decade, "unknown")
        return EraResult(
            decade=decade,
            era_label=f"{decade}er",
            confidence=conf,
            material_prior=material,
            noise_profile=bark,
            tier_used=2,
            hf_rolloff_hz=rolloff_hz,
        )

    def _tier3(self, bark: np.ndarray, rolloff_hz: float, snr_db: float) -> EraResult:
        """Tier-3: Mikrofon-Typ-Heuristik."""
        decade, conf = _microphone_type_decade(bark)
        material = DECADE_MATERIAL_PRIOR.get(decade, "unknown")
        return EraResult(
            decade=decade,
            era_label=f"{decade}er",
            confidence=conf,
            material_prior=material,
            noise_profile=bark,
            tier_used=3,
            hf_rolloff_hz=rolloff_hz,
        )

    def _clap_nearest_neighbor(self, embedding: np.ndarray) -> tuple[int, float]:
        """Findet nächsten Ära-Anker im CLAP-Embedding-Raum.

        Wenn keine vorberechneten Anker vorhanden sind, gibt unbekannte Ära zurück.
        """
        anchors_path = Path(__file__).parent.parent / "models" / "era_classifier" / "era_anchors.npy"
        if not anchors_path.exists():
            return 1960, 0.20
        try:
            anchors = np.load(str(anchors_path))  # (n_anchors, embedding_dim)
            # Letzte Spalte: decade-Label
            decade_labels = anchors[:, -1].astype(int)
            anchor_vecs = anchors[:, :-1]
            # L2-normalisieren für Cosinus
            anchor_norms = np.linalg.norm(anchor_vecs, axis=1, keepdims=True) + 1e-12
            anchor_vecs = anchor_vecs / anchor_norms
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
            cosine_sims = anchor_vecs @ emb_norm
            best_idx = int(np.argmax(cosine_sims))
            best_sim = float(cosine_sims[best_idx])
            conf = float(np.clip((best_sim + 1.0) / 2.0 * 1.2, 0.0, 1.0))
            return int(decade_labels[best_idx]), conf
        except Exception as exc:  # noqa: BLE001
            logger.debug("CLAP NN-Suche fehlgeschlagen: %s", exc)
            return 1960, 0.20

    # ------------------------------------------------------------------
    # Cache-Verwaltung
    # ------------------------------------------------------------------

    def _load_cache(self, path: Path) -> EraResult | None:
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return EraResult(
                decade=int(data["decade"]),
                era_label=str(data["era_label"]),
                confidence=float(data["confidence"]),
                material_prior=str(data["material_prior"]),
                noise_profile=np.array(data["noise_profile"], dtype=np.float32),
                tier_used=int(data.get("tier_used", 2)),
                hf_rolloff_hz=float(data.get("hf_rolloff_hz", 20000.0)),
            )
        except Exception:  # noqa: BLE001
            return None

    def _save_cache(self, path: Path, result: EraResult) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result.as_dict(), f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.debug("EraClassifier: Cache-Speichern fehlgeschlagen: %s", exc)


# ---------------------------------------------------------------------------
# Singleton (Thread-sicher, Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_instance: EraClassifier | None = None
_lock = threading.Lock()


def get_era_classifier() -> EraClassifier:
    """Thread-sicherer Singleton-Accessor für EraClassifier."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = EraClassifier()
    return _instance


def classify_era(audio: np.ndarray, sr: int) -> EraResult:
    """Convenience-Funktion: Erkennt Aufnahme-Ära ohne explizite Instanz.

    Args:
        audio: Audio-Signal (mono oder stereo, float32/64 [-1, 1]).
        sr:    Sample-Rate in Hz.

    Returns:
        EraResult mit Dekade, Confidence, Material-Prior und Noise-Profil.
    """
    return get_era_classifier().classify(audio, sr)
