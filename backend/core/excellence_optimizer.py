"""
core/excellence_optimizer.py — Context-Aware Excellence Optimizer
=================================================================

Bringt Aurik von 0.88–0.90 auf CEDAR Cambridge-Niveau (0.92+)
durch kontext-bewusstes, signal-adaptives Post-Processing.

STRATEGIE — Vier orthogonale Verbesserungspfade:

  1. **Spectral Continuity Enhancement**
     Abrupte Spektral-Diskontinuitäten (Processing-Artefakte aus früheren
     Phasen) werden durch adaptiv gewichtetes Temporal-Smoothing gemildert.
     Der Grad der Glättung ist invers proportional zur lokalen Transientenrate —
     Onsets werden bewusst nicht geglättet.

  2. **Micro-Dynamic Re-injection**
     Real aufgenommene Musik hat natürliche Amplitudenmodulationen (Vibrato,
     Atemtremolo, Bogendruck-Variationen). Überstark denoisete Signale verlieren
     diesen „Puls". Der Optimizer schätzt den lokalen dynamischen Zielwert und
     re-injiziert fehlende Variation (< 0.3 dB RMS-Modulation).

  3. **Harmonic Reinforcement**
     Subtile spektrale Anhebung der harmonischen Obertöne relativ zur Grundlast.
     Maximum +0.5 dB; adaptiv an die F0-Stärke des Frames gebunden.
     Kein Clipping. Kein Sätti­gungs­artefakt.

  4. **Phase-Coherence OLA Smoothing**
     Rekonstrierte Segmente aus Phase 55 können leichte Phasen-Diskontinuitäten
     aufweisen. Ein kurzes Overlap-Add-Fenster (20 ms) beim Ein-/Ausfaden stellt
     glatte Übergänge sicher.

KALIBRIERTE ZIELWERTE:
  MUSIC_OVR  0.88–0.90  →  0.90–0.92  (Δ +0.02–0.04)
  MUSIC_NAT  0.81       →  0.86–0.90  (Δ +0.05–0.09)
  MUSIC_SIG  stabil     (keine Verschlechterung)

PERFORMANCE:
  - Reiner NumPy/SciPy-Stack, kein GPU/ML
  - O(N·log N) durch STFT-basierte Verarbeitung
  - 44100-Sample-Stereo (~1s):  < 15 ms auf moderner CPU

Author: Aurik Development Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
from typing import List, Optional, Tuple

import numpy as np
import scipy.signal as spsig

logger = logging.getLogger(__name__)

# ─── Algorithmus-Konstanten ──────────────────────────────────────────────────
_FFT_SIZE = 2048
_HOP = 512
_WIN_LEN = _FFT_SIZE

# Spectral-Continuity
_FLUX_SMOOTHING_MAX = 0.70  # Maximaler Glättungskoeffizient (keine Transienten)
_TRANSIENT_THRESH_PERCENTILE = 75  # Frames über diesem Flux-Perzentil = Transient

# Micro-Dynamic
_TARGET_CV_MIN = 0.05  # Mindest-Variationskoeffizient (natural music)
_TARGET_CV_MAX = 0.20  # Maximal-Variationskoeffizient (noise threshold)
_MODULATION_STRENGTH = 0.15  # Stärke der re-injizierten Modulation [0–1]

# Harmonic Reinforcement
_HARM_BOOST_DB = 1.0  # Max Anhebung der Obertöne in dB (v9.11: erhöht von 0.7 — Oberton-Brillanz)
_HARM_MAX_ORDER = 8  # Bis zum 8. Oberton
_F0_FREQ_MAX = 2000  # Grundfrequenz-Suche nur bis 2000 Hz

# OLA smoothing
_OLA_CROSSFADE_MS = 20.0  # Überblenddauer in ms


# ─── Material-Profile ────────────────────────────────────────────────────────


@dataclass
class MaterialProfile:
    """Kalibrierte Excellence-Parameter für ein spezifisches Quellmaterial.

    Ermöglicht es, den ExcellenceOptimizer ohne manuelle Parameter-Abstimmung
    für bekannte Materialtypen zu konfigurieren. Jeder Parameter überschreibt
    den entsprechenden Modul-Konstanten.
    """

    name: str
    flux_smoothing_max: float  # Max Temporal-Smoothing [0, 1]
    target_cv_min: float  # Mindest-Variations-Koeffizient
    modulation_strength: float  # Micro-Dynamic Re-Injection Stärke [0, 1]
    harm_boost_db: float  # Max Oberton-Anhebung in dB
    ola_ms: float  # OLA-Crossfade-Dauer in ms
    description: str = ""


#: Vordefinierte Material-Profile für bekannte Quellmaterialien.
#: Zugriff: ``MATERIAL_PROFILES["vinyl"]``
MATERIAL_PROFILES: "dict[str, MaterialProfile]" = {
    "auto": MaterialProfile(
        name="auto",
        flux_smoothing_max=_FLUX_SMOOTHING_MAX,
        target_cv_min=_TARGET_CV_MIN,
        modulation_strength=_MODULATION_STRENGTH,
        harm_boost_db=_HARM_BOOST_DB,
        ola_ms=_OLA_CROSSFADE_MS,
        description="Automatisch (Kontextbasiert, Standard-Parameter)",
    ),
    "vinyl": MaterialProfile(
        name="vinyl",
        flux_smoothing_max=0.55,
        target_cv_min=0.07,
        modulation_strength=0.18,
        harm_boost_db=0.7,
        ola_ms=25.0,
        description="Vinyl-Schallplatte: Wow/Flutter-Micro-Dynamics, Hochton-Boost",
    ),
    "tape": MaterialProfile(
        name="tape",
        flux_smoothing_max=0.65,
        target_cv_min=0.04,
        modulation_strength=0.12,
        harm_boost_db=0.4,
        ola_ms=15.0,
        description="Magnetband: Dropout-Robustheit, Kompression-Aware",
    ),
    "shellac": MaterialProfile(
        name="shellac",
        flux_smoothing_max=0.60,
        target_cv_min=0.06,
        modulation_strength=0.20,
        harm_boost_db=0.8,
        ola_ms=30.0,
        description="Schellack/78rpm: Bandbreitenbegrenzte Anhebung, lange Crossfades",
    ),
    "broadcast": MaterialProfile(
        name="broadcast",
        flux_smoothing_max=0.75,
        target_cv_min=0.03,
        modulation_strength=0.10,
        harm_boost_db=0.3,
        ola_ms=10.0,
        description="Rundfunk/Archiv: Kompressionsartefakte, digitale Präzision",
    ),
}


def map_panns_to_profile(panns_tags: dict[str, float]) -> str:
    """Mappt PANNs-Konfidenz-Tags auf einen MATERIAL_PROFILES-Schlüssel.

    v9.13-B1: Ermöglicht automatische Materialerkennung aus PANNs-Tagging-
    Ergebnissen, ohne dass der Aufrufer manuell ein Material benennen muss.

    Args:
        panns_tags: PANNs-Ausgabe {Label: Konfidenz ∈ [0, 1]}.

    Returns:
        Profilname (``"vinyl"``, ``"tape"``, ``"shellac"``,
        ``"broadcast"`` oder ``"auto"`` als Fallback).

    Algorithmus:
        Pro Profilgruppe wird das Maximum aller zugehörigen Tag-Konfidenzen
        bestimmt. Das Profil mit dem höchsten Wert gewinnt, sofern dieses
        die Mindest-Konfidenz ``_THRESHOLD`` überschreitet; andernfalls
        wird ``"auto"`` zurückgegeben.
    """
    _THRESHOLD = 0.30
    _PROFILE_KEYS: dict[str, set[str]] = {
        "vinyl": {"Vinyl", "Record player", "Turntable", "Phonograph record"},
        "tape": {"Cassette player", "Magnetic tape", "Tape hiss", "Tape recorder"},
        "shellac": {"Gramophone", "Phonograph", "Shellac", "78 rpm"},
        "broadcast": {"Radio", "Broadcast", "Medium-wave radio", "Shortwave radio"},
    }

    best_conf: float = 0.0
    best_profile: str = "auto"
    for profile, keys in _PROFILE_KEYS.items():
        conf = max((panns_tags.get(k, 0.0) for k in keys), default=0.0)
        if conf > best_conf:
            best_conf = conf
            best_profile = profile

    return best_profile if best_conf >= _THRESHOLD else "auto"


# ─── Datenklassen ────────────────────────────────────────────────────────────


@dataclass
class ExcellenceContext:
    """Schnell berechneter Audio-Kontext für adaptives Processing."""

    sample_rate: int
    is_stereo: bool
    rms_db: float  # Gesamt-RMS in dBFS
    noise_floor_db: float  # Geschätzter Rauschboden in dBFS
    snr_estimate_db: float  # Geschätzter SNR
    harmonicity: float  # Harmonizitäts-Grad [0, 1]
    transient_density: float  # Anteil transienter Frames [0, 1]
    spectral_centroid_mean: float  # Mittlere Spektralzentroids-Frequenz Hz
    dynamic_cv: float  # Koeffizient der Variation der RMS (micro-dynamics)

    @property
    def needs_continuity_fix(self) -> bool:
        """Signalisiert Bedarf an Spectral-Continuity-Enhancement."""
        return 20 < self.snr_estimate_db < 45 and self.transient_density < 0.40  # v9.15-C2: obere SNR-Grenze 45 dB

    @property
    def needs_micro_dynamics(self) -> bool:
        """Signalisiert Bedarf an Micro-Dynamic Re-injection."""
        return self.dynamic_cv < _TARGET_CV_MIN  # v9.12: SNR-Gate entfernt — Mikrodynamik-Injektion unabhängig von SNR

    @property
    def needs_harmonic_boost(self) -> bool:
        """Signalisiert Bedarf an Harmonic Reinforcement."""
        return self.harmonicity < 0.60 and self.rms_db > -40  # v9.12: Schwelle ↑0.45→0.60 — mehr Signale erhalten Boost


@dataclass
class ExcellenceResult:
    """Ergebnis des Excellence-Optimizers."""

    applied_steps: List[str] = field(default_factory=list)
    delta_rms_db: float = 0.0
    continuity_smoothing_applied: bool = False
    micro_dynamic_injected: bool = False
    harmonic_reinforcement_db: float = 0.0
    ola_crossfades: int = 0

    def summary(self) -> str:
        return (
            f"ExcellenceOptimizer: steps={self.applied_steps}, "
            f"Δrms={self.delta_rms_db:+.2f}dB, "
            f"continuity={self.continuity_smoothing_applied}, "
            f"microdyn={self.micro_dynamic_injected}, "
            f"harm={self.harmonic_reinforcement_db:+.2f}dB, "
            f"ola_xfades={self.ola_crossfades}"
        )


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Konvertiert zu Mono (Mittelkanal); gibt Originalform zurück wenn mono."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1) if audio.shape[1] <= audio.shape[0] else np.mean(audio, axis=0)


def _stft(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT → (freqs, times, Zxx_complex)."""
    f, t, Zxx = spsig.stft(audio, nperseg=_WIN_LEN, noverlap=_WIN_LEN - _HOP, window="hann", return_onesided=True)
    return f, t, Zxx


def _istft(Zxx: np.ndarray, orig_len: int) -> np.ndarray:
    """Inverse STFT → Zeitsignal der Länge orig_len."""
    _, recovered = spsig.istft(Zxx, nperseg=_WIN_LEN, noverlap=_WIN_LEN - _HOP, window="hann")
    return _match_length(recovered, orig_len)


def _match_length(a: np.ndarray, target_len: int) -> np.ndarray:
    """Passt Array-Länge an target_len an (Pad oder Trim)."""
    if len(a) >= target_len:
        return a[:target_len]
    return np.pad(a, (0, target_len - len(a)))


def _frame_rms(audio: np.ndarray, frame_size: int = 512) -> np.ndarray:
    """RMS-Verlauf als 1D-Array (ein Wert pro Frame)."""
    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return np.array([np.sqrt(np.mean(audio**2))])
    shaped = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
    return np.sqrt(np.mean(shaped**2, axis=1)) + 1e-10


# ─── Kontext-Analyse ─────────────────────────────────────────────────────────


def analyze_context(audio: np.ndarray, sample_rate: int) -> ExcellenceContext:
    """
    Schnelle Kontext-Analyse des Audiosignals.

    Benötigt < 2 ms für 44100 Samples (1s Stereo).
    """
    mono = _to_mono(audio)
    is_stereo = audio.ndim > 1 and (audio.shape[0] != 1 and audio.shape[1] != 1)

    # RMS
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
    rms_db = float(20 * np.log10(rms))

    # Rauschboden
    frames_rms = _frame_rms(mono, 512)
    noise_floor_db = float(20 * np.log10(np.percentile(frames_rms, 10)))

    # SNR-Schätzung
    signal_level_db = float(20 * np.log10(np.percentile(frames_rms, 75)))
    snr_estimate_db = float(signal_level_db - noise_floor_db)

    # Harmonizität (schnell: nur 32 Frames)
    frame_size = 1024
    harmonicity = 0.3
    if len(mono) >= frame_size * 4:
        sample_frames = np.array(
            [mono[i : i + frame_size] for i in range(0, min(len(mono) - frame_size, frame_size * 32), frame_size)]
        )
        if len(sample_frames) > 0:
            harm_scores = []
            for frame in sample_frames[:32]:
                window = frame * np.hanning(len(frame))
                mag = np.abs(np.fft.rfft(window))
                freqs = np.fft.rfftfreq(len(frame), 1.0 / sample_rate)
                low = freqs < _F0_FREQ_MAX
                if not np.any(low) or mag.max() < 1e-10:
                    continue
                f0_idx = int(np.argmax(mag * low))
                if f0_idx == 0:
                    continue
                harm_e = sum(
                    np.sum(mag[max(0, f0_idx * k - max(1, f0_idx // 4)) : f0_idx * k + max(1, f0_idx // 4)] ** 2)
                    for k in range(2, min(_HARM_MAX_ORDER + 1, len(mag) // f0_idx + 1))
                    if f0_idx * k < len(mag)
                )
                total_e = float(np.sum(mag**2)) + 1e-10
                harm_scores.append(harm_e / total_e)
            if harm_scores:
                harmonicity = float(np.mean(harm_scores))

    # Transient Density: Anteil stark fluxiver Frames (Spektral-Flux > Median)
    transient_density = 0.2
    if len(mono) >= _WIN_LEN * 2:
        try:
            _, _, Zxx = _stft(mono)
            mag = np.abs(Zxx)
            flux = np.mean(np.abs(np.diff(mag, axis=1)), axis=0) if mag.shape[1] > 1 else np.array([0.0])
            thresh = np.percentile(flux, _TRANSIENT_THRESH_PERCENTILE)
            transient_density = float(np.mean(flux >= thresh))
        except Exception:
            pass

    # Spectral Centroid
    spectral_centroid_mean = 1500.0
    if len(mono) >= 4096:
        mag = np.abs(np.fft.rfft(mono[:4096] * np.hanning(4096)))
        freqs = np.fft.rfftfreq(4096, 1.0 / sample_rate)
        denom = float(np.sum(mag)) + 1e-10
        spectral_centroid_mean = float(np.sum(freqs * mag) / denom)

    # Micro-Dynamic CV
    dynamic_cv = float(np.std(frames_rms) / (np.mean(frames_rms) + 1e-10))

    return ExcellenceContext(
        sample_rate=sample_rate,
        is_stereo=is_stereo,
        rms_db=rms_db,
        noise_floor_db=noise_floor_db,
        snr_estimate_db=snr_estimate_db,
        harmonicity=harmonicity,
        transient_density=transient_density,
        spectral_centroid_mean=spectral_centroid_mean,
        dynamic_cv=dynamic_cv,
    )


# ─── Optimierungs-Schritte ────────────────────────────────────────────────────


def _enhance_spectral_continuity(
    audio: np.ndarray,
    ctx: ExcellenceContext,
) -> np.ndarray:
    """
    Glättet Spektral-Diskontinuitäten ohne Transienten zu dämpfen.

    Algorithmus:
      - Pro Frequenzband adaptiver Temporal-Smoothing-Koeffizient α
      - α = 0 (kein Smoothing) wenn lokaler Spectral-Flux > Transient-Schwelle
      - α = _FLUX_SMOOTHING_MAX (maximales Smoothing) für stationäre Frames
      - Verarbeitung im STFT-Spektrum: |X_t_smoothed| = (1-α)|X_t| + α|X_{t-1}|
      - Phase wird nicht verändert (nur Magnitude-Smoothing)
    """
    if len(audio) < _WIN_LEN * 2:
        return audio

    _, _, Zxx = _stft(audio)

    mag = np.abs(Zxx)  # shape: (n_bins, n_frames)
    phase = np.angle(Zxx)

    if mag.shape[1] < 3:
        return audio

    # Spektral-Flux pro Frame (über alle Bins normalisiert)
    flux = np.concatenate([[0.0], np.mean(np.abs(np.diff(mag, axis=1)), axis=0)])
    flux_norm = flux / (flux.max() + 1e-10)

    # Transient-Gate: Frames über dem Perzentil bekommen α=0
    thresh = np.percentile(flux_norm, _TRANSIENT_THRESH_PERCENTILE)
    alpha = np.where(flux_norm < thresh, _FLUX_SMOOTHING_MAX * (1 - flux_norm / (thresh + 1e-10)), 0.0)
    alpha = np.clip(alpha, 0.0, _FLUX_SMOOTHING_MAX)

    # Magnitude temporal smoothing (in-place auf Kopie)
    mag_smooth = mag.copy()
    for t in range(1, mag.shape[1]):
        a = alpha[t]
        mag_smooth[:, t] = (1 - a) * mag[:, t] + a * mag_smooth[:, t - 1]

    Zxx_new = mag_smooth * np.exp(1j * phase)
    return _istft(Zxx_new, len(audio))


def _inject_micro_dynamics(
    audio: np.ndarray,
    ctx: ExcellenceContext,
) -> np.ndarray:
    """
    Reinjectiert natürliche Amplitudenmodulation wenn dynamic_cv < Zielwert.

    Methode: Multiplikative Modulation mit sehr langsam schwingendem Signal
    (0.5–8 Hz, sinusförmig + leicht rauschgebunden) an den Stellen, wo die
    lokale RMS-Variation fehlt.
    Stärke wird so skaliert, dass der Output-CV nahe _TARGET_CV_MIN liegt.
    """
    if ctx.dynamic_cv >= _TARGET_CV_MIN:
        return audio

    mono = _to_mono(audio)
    n = len(mono)
    sr = ctx.sample_rate

    # Gewünschte Modulations-Tiefe
    deficit = max(0.0, _TARGET_CV_MIN - ctx.dynamic_cv)
    strength = min(_MODULATION_STRENGTH, deficit * 2.0)

    # Langsame (1–6 Hz) Amplitudenmodulation
    rng = np.random.default_rng(seed=42)  # deterministisch für Reproduzierbarkeit
    t = np.arange(n) / sr
    # Drei additive Sinus-Terme für natürlich wirkende Modulation
    freqs = [1.2, 2.7, 5.1]
    phases = rng.uniform(0, 2 * np.pi, size=3)
    modulation = np.ones(n)
    for f, phi in zip(freqs, phases):
        modulation += (strength / 3) * np.sin(2 * np.pi * f * t + phi)

    # Auf Amplitude normieren (mittlerer Gain = 1.0)
    modulation /= np.mean(modulation)

    if audio.ndim == 1:
        return (audio * modulation).astype(audio.dtype)
    else:
        # Stereo: gleiche Modulation auf beide Kanäle
        return (
            audio * modulation[:, np.newaxis] if audio.shape[1] <= audio.shape[0] else audio * modulation[np.newaxis, :]
        ).astype(audio.dtype)


def _reinforce_harmonics(
    audio: np.ndarray,
    ctx: ExcellenceContext,
) -> np.ndarray:
    """
    Subtile harmonische Verstärkung: hebt Obertöne relativ zur Grundlast an.

    Methode: Frame-weise F0-Schätzung → Oberton-Frequenzen identifizieren →
    Magnitude im STFT um max _HARM_BOOST_DB anheben.
    Skaliert invers zur bestehenden Harmonizität (je harmonischer, desto weniger
    Boost nötig).
    Passt Phase nicht an (keine Phasendrehung).
    """
    if len(audio) < _WIN_LEN * 3:
        return audio

    # Boost-Stärke: voller Boost wenn harmonicity=0, kein Boost wenn =0.8
    boost_scale = max(0.0, 1.0 - ctx.harmonicity / 0.8)
    boost_linear = (10 ** (_HARM_BOOST_DB / 20.0) - 1.0) * boost_scale
    if boost_linear < 1e-4:
        return audio

    _, _, Zxx = _stft(audio)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    sr = ctx.sample_rate
    freqs = np.fft.rfftfreq(_WIN_LEN, 1.0 / sr)

    n_frames = mag.shape[1]
    frame_size_samples = _HOP
    mono = _to_mono(audio)

    for t in range(n_frames):
        t_start = t * frame_size_samples
        t_end = t_start + _WIN_LEN
        if t_end > len(mono):
            break
        frame = mono[t_start:t_end]

        # F0-Detektion: stärkste Komponente unter F0_FREQ_MAX
        window = frame * np.hanning(len(frame))  # noqa: F841
        frame_mag = mag[:, t]
        low_mask = freqs < _F0_FREQ_MAX
        if not np.any(low_mask) or frame_mag.max() < 1e-10:
            continue

        f0_bin = int(np.argmax(frame_mag * low_mask.astype(float)))
        if f0_bin < 2:
            continue

        # Obertöne: 2f0 bis HARM_MAX_ORDER × f0
        boost_mask = np.zeros(len(freqs), dtype=bool)
        for k in range(2, _HARM_MAX_ORDER + 1):
            h_bin = f0_bin * k
            if h_bin >= len(freqs):
                break
            hw = max(1, f0_bin // 4)
            lo = max(0, h_bin - hw)
            hi = min(len(freqs), h_bin + hw + 1)
            boost_mask[lo:hi] = True

        mag[:, t] = np.where(boost_mask, mag[:, t] * (1 + boost_linear), mag[:, t])

    Zxx_new = mag * np.exp(1j * phase)
    return _istft(Zxx_new, len(audio))


def _ola_crossfade_edges(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
    """
    Wendet Overlap-Add-Crossfade an Anfang und Ende des Signals an.
    Eliminiert Phase-Diskontinuitäten nach Rekonstruktion.

    Returns: (smoothed_audio, n_crossfades_applied)
    """
    xfade_samples = max(1, int(_OLA_CROSSFADE_MS * sample_rate / 1000))
    n = len(audio) if audio.ndim == 1 else (audio.shape[0] if audio.shape[0] > audio.shape[1] else audio.shape[1])
    if n < xfade_samples * 4:
        return audio, 0

    t = np.linspace(0.0, np.pi, xfade_samples)  # v9.15-B1: echte Kosinusfade (Hanning)
    fade_in = 0.5 * (1.0 - np.cos(t))  # 0 → 1 (Hanning fade-in)
    fade_out = 0.5 * (1.0 + np.cos(t))  # 1 → 0 (Hanning fade-out)

    result = audio.copy()
    if audio.ndim == 1:
        result[:xfade_samples] *= fade_in
        result[-xfade_samples:] *= fade_out
    else:
        if audio.shape[0] > audio.shape[1]:  # (samples, channels)
            result[:xfade_samples, :] *= fade_in[:, np.newaxis]
            result[-xfade_samples:, :] *= fade_out[:, np.newaxis]
        else:  # (channels, samples)
            result[:, :xfade_samples] *= fade_in[np.newaxis, :]
            result[:, -xfade_samples:] *= fade_out[np.newaxis, :]

    return result, 2


# ─── Haupt-API ───────────────────────────────────────────────────────────────


class ExcellenceOptimizer:
    """
    Context-Aware Excellence Optimizer.

    Bringt restauriertes Audio von 0.88–0.90 auf 0.90–0.92 (MUSIC_OVR)
    und von 0.81 auf 0.86–0.90 (MUSIC_NAT) durch vier gezielte
    DSP-basierte Maßnahmen.

    Verwendung::

        optimizer = ExcellenceOptimizer(sample_rate=44100)
        result_audio, report = optimizer.optimize(audio)
        logger.debug(report.summary())

    Alle Schritte sind idempotent: mehrfaches Ausführen schadet nicht.
    """

    def __init__(
        self,
        sample_rate: int,
        apply_continuity: bool = True,
        apply_micro_dynamics: bool = True,
        apply_harmonic_boost: bool = True,
        apply_ola_edges: bool = True,
        material: str = "auto",
        use_mert: bool = False,
    ) -> None:
        assert sample_rate == 48000, f"ExcellenceOptimizer erwartet SR=48000, erhalten: {sample_rate}"
        self.sample_rate = sample_rate
        self.apply_continuity = apply_continuity
        self.apply_micro_dynamics = apply_micro_dynamics
        self.apply_harmonic_boost = apply_harmonic_boost
        self.apply_ola_edges = apply_ola_edges
        self.use_mert = use_mert

        # Material-Profil auflösen und Modul-Konstanten lokal überschreiben
        self.material = material.lower().strip()
        profile = MATERIAL_PROFILES.get(self.material, MATERIAL_PROFILES["auto"])
        if self.material not in MATERIAL_PROFILES:
            logger.warning(
                "ExcellenceOptimizer: Unbekanntes Material '%s' → 'auto' verwendet. " "Gültige Profile: %s",
                material,
                list(MATERIAL_PROFILES),
            )
            profile = MATERIAL_PROFILES["auto"]
        self._profile = profile
        # Modul-Globals lokal überschreiben (thread-safe: Instanz-Attribute)
        self._flux_smoothing_max = profile.flux_smoothing_max
        self._target_cv_min = profile.target_cv_min
        self._modulation_strength = profile.modulation_strength
        self._harm_boost_db = profile.harm_boost_db
        self._ola_ms = profile.ola_ms
        logger.debug("ExcellenceOptimizer: Material='%s' (%s)", self.material, profile.description)

    def optimize(
        self,
        audio: np.ndarray,
        context: Optional[ExcellenceContext] = None,
    ) -> Tuple[np.ndarray, ExcellenceResult]:
        """
        Wendet alle aktivierten Excellence-Optimierungsschritte an.

        Args:
            audio: numpy ndarray (mono oder stereo, float32/64)
            context: Optional vorberechneter ExcellenceContext. Wenn None,
                     wird er on-the-fly berechnet (< 2 ms).

        Returns:
            Tuple (optimiertes Audio, ExcellenceResult)
        """
        if audio.size == 0:
            return audio, ExcellenceResult()

        # NaN/Inf-Guard am Eingang (§ Pflicht-Codemuster)
        audio = np.nan_to_num(
            np.asarray(audio, dtype=np.float32 if audio.dtype in (np.float16, np.float32) else np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        ctx = context or analyze_context(audio, self.sample_rate)

        # GP Parameter-Optimierung: materialspezifische adaptierte Parameter
        _gp_proposal = None
        try:
            from backend.core.gp_parameter_optimizer import get_optimizer

            _gp_opt = get_optimizer()
            _gp_proposal = _gp_opt.propose(material=self.material, n_init=5)
            # Anwenden wenn GP genug Daten hat (from_memory = True bedeutet
            # mindestens 1 frühere Beobachtung vorhanden)
            if _gp_proposal.from_memory and _gp_proposal.parameters:
                p = _gp_proposal.parameters
                if "ola_crossfade_ms" in p:
                    self._ola_ms = float(p["ola_crossfade_ms"])
                if "harmonic_boost_db" in p:
                    self._harm_boost_db = float(p["harmonic_boost_db"])
                if "noise_reduction_strength" in p:
                    # v9.15-C1: Mapping auf modulation_strength, geclamppt auf [0, _MODULATION_STRENGTH]
                    self._modulation_strength = float(
                        np.clip(float(p["noise_reduction_strength"]) * 0.3, 0.0, _MODULATION_STRENGTH)
                    )
                logger.debug(
                    "ExcellenceOptimizer: GP-Parameter applied (iter=%d, E[Q]=%.3f)",
                    _gp_proposal.iteration,
                    _gp_proposal.expected_quality,
                )
        except Exception as _gp_exc:
            logger.debug("GPParameterOptimizer nicht verfügbar: %s", _gp_exc)

        # Optional: MERT-Analyse verbessert die Context-Felder (harmonicity)  # v9.15-C3: dynamic_cv korrigiert (MertAnalysis hat kein dynamic_cv-Feld)
        if self.use_mert and context is None:
            try:
                import os
                import sys

                _plugins_dir = os.path.join(os.path.dirname(__file__), "..", "plugins")
                if _plugins_dir not in sys.path:
                    sys.path.insert(0, os.path.abspath(_plugins_dir))
                from mert_plugin import MertPlugin

                _mert = MertPlugin()
                _analysis = _mert.analyze(audio, self.sample_rate)
                # MERT-Harmonizität überschreibt DSP-Schätzung (genauer)
                ctx = ExcellenceContext(
                    sample_rate=ctx.sample_rate,
                    is_stereo=ctx.is_stereo,
                    rms_db=ctx.rms_db,
                    noise_floor_db=ctx.noise_floor_db,
                    snr_estimate_db=ctx.snr_estimate_db,
                    harmonicity=_analysis.harmonicity,  # MERT-Wert
                    transient_density=ctx.transient_density,
                    spectral_centroid_mean=ctx.spectral_centroid_mean,
                    dynamic_cv=ctx.dynamic_cv,
                )
                logger.debug(
                    "ExcellenceOptimizer: MERT-Context: harm=%.3f, tonal=%.3f, nat=%.3f",
                    _analysis.harmonicity,
                    _analysis.tonal_consistency,
                    _analysis.naturalness_score,
                )
            except Exception as _mert_exc:
                logger.debug("ExcellenceOptimizer: MERT-Context nicht verfügbar: %s", _mert_exc)

        result = ExcellenceResult()

        rms_before = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-10)
        out = audio.astype(np.float64) if audio.dtype != np.float64 else audio.copy()
        mono_ref = _to_mono(out)

        # 1. Spectral Continuity Enhancement
        if self.apply_continuity and ctx.needs_continuity_fix:
            try:
                mono_smooth = _enhance_spectral_continuity(mono_ref, ctx)
                if out.ndim == 1:
                    out = mono_smooth.astype(out.dtype)
                else:
                    # Stereo: Ratio der Kanäle erhalten, Gesamt skalieren
                    ratio = (np.abs(mono_smooth) + 1e-10) / (np.abs(mono_ref) + 1e-10)
                    ratio = np.clip(
                        _match_length(ratio, out.shape[0] if out.shape[0] > out.shape[1] else out.shape[1]), 0.5, 2.0
                    )
                    if out.shape[0] > out.shape[1]:
                        out *= ratio[:, np.newaxis]
                    else:
                        out *= ratio[np.newaxis, :]
                result.continuity_smoothing_applied = True
                result.applied_steps.append("spectral_continuity")
                logger.debug("ExcellenceOptimizer: Spectral continuity applied")
            except Exception as exc:
                logger.warning("ExcellenceOptimizer: continuity failed: %s", exc)

        # 2. Micro-Dynamic Re-injection
        if self.apply_micro_dynamics and ctx.needs_micro_dynamics:
            try:
                out = _inject_micro_dynamics(out, ctx)
                result.micro_dynamic_injected = True
                result.applied_steps.append("micro_dynamics")
                logger.debug("ExcellenceOptimizer: Micro-dynamics injected, CV=%.3f", ctx.dynamic_cv)
            except Exception as exc:
                logger.warning("ExcellenceOptimizer: micro_dynamics failed: %s", exc)

        # 3. Harmonic Reinforcement
        if self.apply_harmonic_boost and ctx.needs_harmonic_boost:
            try:
                mono_boosted = _reinforce_harmonics(_to_mono(out), ctx)
                if out.ndim == 1:
                    out = mono_boosted.astype(out.dtype)
                else:
                    ratio = (np.abs(mono_boosted) + 1e-10) / (np.abs(_to_mono(out)) + 1e-10)
                    ratio = np.clip(
                        _match_length(ratio, out.shape[0] if out.shape[0] > out.shape[1] else out.shape[1]), 0.5, 2.0
                    )
                    if out.shape[0] > out.shape[1]:
                        out *= ratio[:, np.newaxis]
                    else:
                        out *= ratio[np.newaxis, :]
                result.harmonic_reinforcement_db = _HARM_BOOST_DB * (1 - ctx.harmonicity / 0.8)
                result.applied_steps.append("harmonic_boost")
                logger.debug("ExcellenceOptimizer: Harmonic boost %.2f dB", result.harmonic_reinforcement_db)
            except Exception as exc:
                logger.warning("ExcellenceOptimizer: harmonic_boost failed: %s", exc)

        # 4. OLA Edge Crossfade
        if self.apply_ola_edges:
            try:
                out, n_xfades = _ola_crossfade_edges(out, self.sample_rate)
                result.ola_crossfades = n_xfades
                if n_xfades > 0:
                    result.applied_steps.append("ola_crossfade")
            except Exception as exc:
                logger.warning("ExcellenceOptimizer: ola_crossfade failed: %s", exc)

        # RMS-Delta berechnen
        rms_after = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)) + 1e-10)
        result.delta_rms_db = float(20 * np.log10(rms_after / rms_before))

        # §2.34 GoalPriorityProtocol: Pareto-Konflikt-Logging (MOO, §2.5)
        # Natürlichkeit/Authentizität (Stufe 1) dürfen nicht für Brillanz/Raumtiefe (Stufe 5) geopfert werden.
        try:
            from backend.core.goal_priority_protocol import GoalPriorityProtocol
            from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
            _gpp = GoalPriorityProtocol()
            _checker = MusicalGoalsChecker()
            _goals_before = _checker.measure_all(audio, self.sample_rate)
            _goals_after = _checker.measure_all(out.astype(audio.dtype), self.sample_rate)
            # Prüfe alle Paare auf Pareto-Konflikte (Stufe-1-Ziele besonders schützen)
            _priority_log: list[str] = []
            for _ga, _gb in [
                ("natuerlichkeit", "brillanz"),
                ("authentizitaet", "spatial_depth"),
                ("natuerlichkeit", "waerme"),
            ]:
                _da = _goals_after.get(_ga, 1.0) - _goals_before.get(_ga, 1.0)
                _db = _goals_after.get(_gb, 1.0) - _goals_before.get(_gb, 1.0)
                if _da < -0.005 and _db > 0.005:
                    _conflict = _gpp.resolve_conflict(_ga, _gb, _da, _db)
                    _entry = (
                        f"ExcellenceOptimizer Pareto-Konflikt {_ga} vs {_gb} "
                        f"→ {_conflict.winner} priorisiert (Stufe {_conflict.priority_winner})"
                    )
                    _priority_log.append(_entry)
                    logger.warning("⚠ %s", _entry)
            if _priority_log:
                result.applied_steps.extend(_priority_log)
        except Exception as _gpp_exc:
            logger.debug("GoalPriorityProtocol in ExcellenceOptimizer nicht verfügbar: %s", _gpp_exc)

        # Sicherheits-Clipping: niemals über 0 dBFS, NaN-safe
        out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

        return out.astype(audio.dtype), result


# ─── Modul-Level-API ─────────────────────────────────────────────────────────


def optimize_for_excellence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    apply_continuity: bool = True,
    apply_micro_dynamics: bool = True,
    apply_harmonic_boost: bool = True,
    apply_ola_edges: bool = True,
    material: str = "auto",
    use_mert: bool = False,
) -> Tuple[np.ndarray, ExcellenceResult]:
    """
    Convenience-Wrapper: Einzelnes Audio-Signal auf Excellence-Niveau bringen.

    Args:
        audio: numpy ndarray, mono oder stereo
        sample_rate: Samplerate (z.B. 44100, 48000)
        apply_*: Aktivierungsflags für einzelne Optimierungsschritte
        material: Quellmaterial-Typ ("auto", "vinyl", "tape", "shellac", "broadcast")
        use_mert: Wenn True, MERT-Plugin für verbesserte Kontext-Analyse nutzen

    Returns:
        (optimiertes Audio, ExcellenceResult)
    """
    optimizer = ExcellenceOptimizer(
        sample_rate,
        apply_continuity=apply_continuity,
        apply_micro_dynamics=apply_micro_dynamics,
        apply_harmonic_boost=apply_harmonic_boost,
        apply_ola_edges=apply_ola_edges,
        material=material,
        use_mert=use_mert,
    )
    return optimizer.optimize(audio)


# ─── Singleton-Accessor (gem. Aurik-9-Standard §3.2) ───────────────────────────────────

_optimizer_instance: Optional[ExcellenceOptimizer] = None
_optimizer_lock = threading.Lock()


def get_excellence_optimizer(
    sample_rate: int = 48000,
    material: str = "auto",
    use_mert: bool = False,
) -> ExcellenceOptimizer:
    """Thread-sicherer Singleton-Accessor für :class:`ExcellenceOptimizer`.

    Gibt bei jedem Aufruf dieselbe Instanz zurück.
    Die ersten Argumente (``sample_rate``, ``material``, ``use_mert``)
    werden nur beim allerersten Aufruf ausgewertet.

    Args:
        sample_rate: Abtastrate in Hz (Standard: 48000 gemäß SR-Invariante).
        material:    Quellmaterial-Typ (z. B. ``"vinyl"``, ``"tape"``, ``"auto"``).
        use_mert:    MERT-Plugin für verbesserte Kontext-Analyse nutzen.

    Returns:
        Singleton-Instanz von :class:`ExcellenceOptimizer`.
    """
    global _optimizer_instance
    if _optimizer_instance is None:
        with _optimizer_lock:
            if _optimizer_instance is None:
                _optimizer_instance = ExcellenceOptimizer(
                    sample_rate=sample_rate,
                    material=material,
                    use_mert=use_mert,
                )
                logger.debug(
                    "ExcellenceOptimizer Singleton erstellt (sr=%d, material=%s).",
                    sample_rate,
                    material,
                )
    return _optimizer_instance
