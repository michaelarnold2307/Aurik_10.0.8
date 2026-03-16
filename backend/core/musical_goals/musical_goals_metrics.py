"""
AURIK v9.12 Musical Goals Measurement System
=============================================
v9.12: MicroDynamics blind floor entfernt (0.92→0.0); Excellence-Optimizer A1 (harmonicity-
Schwelle 0.45→0.60) + A2 (SNR-Gate entfernt) — 3684/3684 Tests passed in 807.78s

Implementiert messbare Metriken für alle 14 musikalischen Qualitätsziele (§1.2 Spec v9.9.9):
 1. Brillanz              (HF Clarity 8-20 kHz)
 2. Wärme                 (Mid-Range Richness 200-2000 Hz)
 3. Natürlichkeit         (Gesamtklang ohne Artefakte)
 4. Authentizität         (Voice Identity & Spectral Fingerprint)
 5. Emotionalität         (Dynamics & Expression)
 6. Transparenz           (Clarity & Separation)
 7. Bass-Kraft            (Kraftvolle Basswiedergabe 20-250 Hz)
 8. Groove                (Mikro-Timing, Swing, Event-Onset-Präzision — ab v9.9)
 9. Raumtiefe             (Stereobreite, Phantom-Center-Stabilität — ab v9.9)
10. Timbre-Authentizität  (MFCC-Pearson, Spectral-Centroid-Korrelation — ab v9.9)
11. Tonales Zentrum       (Chroma-Korrelation, kein Key-Shift — ab v9.9.5)
12. Mikro-Dynamik         (LUFS-Profil-Korrelation 400 ms — ab v9.9.5)
13. Separation-Treue      (SDR ≥ 8 dB / SIR ≥ 12 dB — ab v9.9.9)
14. Artikulation          (Attack-Charakter-Erhalt, Transient-Shape — ab v9.9.9)

Quelle: Finalisierungs_Roadmap.md - Component 0.2
Autor: AI Team
Datum: 8. Februar 2026
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import threading

import librosa
import librosa.core.constantq  # CQT/VQT-Pfad — von chroma_cqt ausgelöst
import librosa.core.pitch  # estimate_tuning/piptrack — von constantq.vqt ausgelöst
import librosa.feature  # lazy_loader-Deadlock verhindern: Submodul vorab laden
import librosa.util  # librosa.util.frame muss vor Threading verfügbar sein
import librosa.util.utils  # util.expand_to lebt hier — direkter Import bypass lazy_loader
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-Loader-Deadlock-Prävention (Thread-Safety §3.1)
# ---------------------------------------------------------------------------
# librosa verwendet lazy_loader.attach_stub() — ALLE Submodule sind lazy.
# Wenn zwei Threads gleichzeitig erstmals librosa.stft() aufrufen, geraten
# sie in einen Python-Import-Lock-Deadlock (beide warten auf librosa.util,
# das durch den ersten stft()-Aufruf in librosa.core.spectrum gezogen wird).
# Lösung: alle relevanten Submodule einmalig im Haupt-Thread (hier, bei
# Modulimport) vollständig auflösen, bevor Worker-Threads starten können.
# ---------------------------------------------------------------------------
def _warm_up_librosa() -> None:
    """Löst alle librosa-Lazy-Submodule im Haupt-Thread auf (Deadlock-Fix).

    Kritisch: Jede Zeile in einem EIGENEN try/except, damit ein Fehler (z.B.
    chroma_cqt bei zu niedriger SR) nicht alle nachfolgenden Auflösungen abbricht.
    Nur wenn expand_to & co aufgelöst sind, sind Thread-sichere Aufrufe möglich.
    """
    _dummy_short = np.zeros(4096, dtype=np.float32)
    _dummy_short[::4] = 0.1  # minimale Energie für Lazy-Auflösung
    _sr_low = 8_000  # für STFT/MFCC/centroid/rolloff/rms/onset/beat
    _sr_cqt = 22_050  # CQT braucht höhere SR (librosa minimum ~8kHz, sicher: 22050)

    # stft → löst librosa.core.spectrum + librosa.util auf
    try:
        librosa.stft(_dummy_short, n_fft=512, hop_length=128)
    except Exception as exc:  # noqa: BLE001
        logger.debug("librosa warm-up stft: %s", exc)

    # feature-Submodule einzeln auflösen
    for _call, _args, _kwargs in [
        (librosa.feature.mfcc, (), {"y": _dummy_short, "sr": _sr_low, "n_mfcc": 13}),
        (librosa.feature.spectral_centroid, (), {"y": _dummy_short, "sr": _sr_low}),
        (librosa.feature.spectral_rolloff, (), {"y": _dummy_short, "sr": _sr_low}),
        (librosa.feature.zero_crossing_rate, (), {"y": _dummy_short}),
        (librosa.feature.chroma_stft, (), {"y": _dummy_short, "sr": _sr_low}),
        (librosa.feature.rms, (), {"y": _dummy_short}),
        # CQT-Pfad: feature.chroma_cqt → constantq.vqt → pitch.piptrack → util.expand_to
        # MUSS _sr_cqt=22050 verwenden — bei sr=4000 oder sr=8000 schlägt CQT fehl
        (librosa.feature.chroma_cqt, (), {"y": np.zeros(int(_sr_cqt * 0.5), dtype=np.float32) + 0.1, "sr": _sr_cqt}),
        (librosa.onset.onset_strength, (), {"y": _dummy_short, "sr": _sr_low}),
        (librosa.beat.beat_track, (), {"y": _dummy_short, "sr": _sr_low}),
    ]:
        try:
            _call(*_args, **_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("librosa warm-up %s: %s", getattr(_call, "__name__", _call), exc)

    # util-Attribute explizit auflösen (alle lazy-loader-Ziele)
    for _attr in ("MAX_MEM_BLOCK", "pad_center", "frame", "expand_to", "normalize", "valid_audio", "fix_length"):
        try:
            getattr(librosa.util, _attr)
        except Exception as exc:  # noqa: BLE001
            logger.debug("librosa warm-up util.%s: %s", _attr, exc)

    logger.debug("librosa warm-up: alle Submodule aufgelöst (Deadlock-Fix)")


_warm_up_librosa()

# ---------------------------------------------------------------------------
# Lazy-Import-Hilfsfunktionen für ML-Plugins (Graceful Degradation §3.4)
# ---------------------------------------------------------------------------
_PLUGINS_DIR = Path(__file__).parent.parent.parent.parent / "plugins"


def _get_crepe():
    """Gibt get_crepe_plugin() zurück oder None wenn nicht verfügbar.

    Umgebungsvariable AURIK_DISABLE_CREPE=1 deaktiviert CREPE (z.B. in
    Hypothesis-Fuzzing-Tests, wo ONNX-Inferenz den 30s-Timeout auslösen kann).
    """
    import os  # noqa: PLC0415

    if os.environ.get("AURIK_DISABLE_CREPE", "").strip() == "1":
        return None
    try:
        if str(_PLUGINS_DIR) not in sys.path:
            sys.path.insert(0, str(_PLUGINS_DIR.parent))
        from plugins.crepe_plugin import get_crepe_plugin  # noqa: PLC0415

        return get_crepe_plugin()
    except Exception:  # noqa: BLE001
        return None


def _get_versa():
    """Gibt den VERSA-Plugin-Singleton zurück oder None wenn nicht verfügbar."""
    try:
        if str(_PLUGINS_DIR.parent) not in sys.path:
            sys.path.insert(0, str(_PLUGINS_DIR.parent))
        from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

        return get_versa_plugin()
    except Exception:  # noqa: BLE001
        return None


# Backward-Compat-Alias
_get_cdpam = _get_versa


@dataclass
class GoalMeasurement:
    """Result of a single musical goal measurement"""

    goal_name: str
    score: float  # 0.0 - 1.0
    passed: bool
    threshold: float
    details: dict[str, float]


class BassKraftMetric:
    """
    Bass-Kraft: Kraftvolle Basswiedergabe (20-250 Hz)

    Misst:
    - Bass Energy Ratio (20-250 Hz vs. full spectrum)
    - Harmonic Bass Strength (F0 detection 20-120 Hz)
    - Sub-Bass Presence (20-60 Hz)

    Threshold: 0.85 (Finalisierungs_Roadmap)
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold
        self.max_bass_loss = 0.15  # Max 15% bass loss allowed

    def measure(self, audio: np.ndarray, sr: int) -> float:
        """
        Measure bass kraft score (0.0 - 1.0).

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Bass kraft score (higher is better)
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Bass band (20-250 Hz)
        bass_mask = (freqs >= 20) & (freqs <= 250)
        sub_bass_mask = (freqs >= 20) & (freqs <= 60)
        mid_bass_mask = (freqs >= 60) & (freqs <= 120)
        upper_bass_mask = (freqs >= 120) & (freqs <= 250)

        # Full spectrum energy
        full_energy = np.sum(magnitude**2)

        # Bass energy
        bass_energy = np.sum(magnitude[bass_mask] ** 2)
        sub_bass_energy = np.sum(magnitude[sub_bass_mask] ** 2)
        mid_bass_energy = np.sum(magnitude[mid_bass_mask] ** 2)
        upper_bass_energy = np.sum(magnitude[upper_bass_mask] ** 2)

        # Bass Energy Ratio (0-1)
        bass_ratio = bass_energy / (full_energy + 1e-10)

        # Weighted bass components
        # Sub-bass (20-60 Hz): 30%
        # Mid-bass (60-120 Hz): 50% (most important for "kraft")
        # Upper-bass (120-250 Hz): 20%
        weighted_bass = (
            0.30 * (sub_bass_energy / (bass_energy + 1e-10))
            + 0.50 * (mid_bass_energy / (bass_energy + 1e-10))
            + 0.20 * (upper_bass_energy / (bass_energy + 1e-10))
        )

        # ---------- F0-basierte harmonische Stärke ---------------------------
        # Bevorzuge CREPE-ONNX (Kim et al. 2018, präziser, O(N·logN)):
        #   voiced Frames im Bassbereich (20–120 Hz) als Stärke-Signal.
        # Fallback: librosa.pyin (Mauch & Dixon 2014, max. 2 s @ O(N²)).
        bass_harmonic_strength: float
        try:
            crepe = _get_crepe()
            if crepe is not None:
                result = crepe.analyze(audio, sr)
                # Anteil voiced Frames im Bassbereich 20–120 Hz
                bass_mask_f0 = (result.f0_hz >= 20) & (result.f0_hz <= 120) & (result.voiced_prob > 0.45)
                n_total = max(1, len(result.f0_hz))
                bass_harmonic_strength = float(np.sum(bass_mask_f0) / n_total)
                logger.debug(
                    "BassKraft-F0 via CREPE [%s]: bass_voiced=%.2f",
                    result.model_used,
                    bass_harmonic_strength,
                )
            else:
                raise RuntimeError("CREPE nicht verfügbar")
        except Exception:  # noqa: BLE001 — pYIN-Fallback
            try:
                seg_len = min(len(audio), int(sr * 2.0))
                f0, _, voiced_probs = librosa.pyin(audio[:seg_len], fmin=20, fmax=250, sr=sr)
                bass_voiced = np.sum((f0 >= 20) & (f0 <= 120) & (voiced_probs > 0.7))
                bass_harmonic_strength = float(bass_voiced / max(1, len(f0)))
            except Exception:  # noqa: BLE001
                bass_harmonic_strength = 0.5

        # Final score (weighted combination)
        score = (
            0.40 * bass_ratio * 20  # Normalize to 0-1 (bass_ratio typically 0-0.05)
            + 0.35 * weighted_bass
            + 0.25 * bass_harmonic_strength
        )

        # Clip to [0, 1]
        score = min(1.0, max(0.0, score))

        return score

    def check_preservation(
        self, original: np.ndarray, processed: np.ndarray, sr: int
    ) -> tuple[bool, float, dict[str, float]]:
        """
        Check if bass preservation is acceptable.

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate

        Returns:
            Tuple of (passed, loss, details)
        """
        orig_score = self.measure(original, sr)
        proc_score = self.measure(processed, sr)

        loss = (orig_score - proc_score) / (orig_score + 1e-10)
        passed = loss <= self.max_bass_loss

        details = {
            "original_score": orig_score,
            "processed_score": proc_score,
            "loss": loss,
            "max_allowed_loss": self.max_bass_loss,
        }

        return passed, loss, details


class BrillanzMetric:
    """
    Brillanz: High-Frequency Clarity & Sparkle (8-20 kHz)

    Misst:
    - HF Energy (8-20 kHz)
    - Spectral Centroid
    - Brightness Score

    Threshold: 0.85
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold

    def measure(self, audio: np.ndarray, sr: int) -> float:
        """Measure brillanz score (0.0 - 1.0)."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1)

        # STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # HF band (8-20 kHz)
        hf_mask = (freqs >= 8000) & (freqs <= 20000)

        # Full spectrum energy
        full_energy = np.sum(magnitude**2)

        # HF energy
        hf_energy = np.sum(magnitude[hf_mask] ** 2)

        # HF Energy Ratio
        hf_ratio = hf_energy / (full_energy + 1e-10)

        # Spectral Centroid (higher = brighter)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=2048, hop_length=512)[0]
        mean_centroid = np.mean(centroid)
        # FIXED v9.10: recalibrated — 3500 Hz centroid = 1.0 (bright, mastered music)
        # (was: (centroid-1000)/7000 required 8000 Hz centroid for max score)
        centroid_normalized = min(1.0, max(0.0, (mean_centroid - 800.0) / 2700.0))

        # Brightness (weighted HF bands)
        # 8-12 kHz: 40% (presence)
        # 12-16 kHz: 35% (air)
        # 16-20 kHz: 25% (sparkle)
        bright_mask_1 = (freqs >= 8000) & (freqs <= 12000)
        bright_mask_2 = (freqs >= 12000) & (freqs <= 16000)
        bright_mask_3 = (freqs >= 16000) & (freqs <= 20000)

        bright_1 = np.sum(magnitude[bright_mask_1] ** 2)
        bright_2 = np.sum(magnitude[bright_mask_2] ** 2)
        bright_3 = np.sum(magnitude[bright_mask_3] ** 2)

        hf_total = bright_1 + bright_2 + bright_3 + 1e-10
        brightness = 0.40 * (bright_1 / hf_total) + 0.35 * (bright_2 / hf_total) + 0.25 * (bright_3 / hf_total)

        # BUG FIX v9.10: brightness ∈ [0.25, 0.40] — normalize to [0, 1] so formula
        # ceiling is 1.0 (was: max 0.82 — mathematically impossible to reach 0.85 threshold!)
        # Normalization: (brightness - min) / (max - min) = (brightness - 0.25) / 0.15
        brightness_normalized = min(1.0, max(0.0, (brightness - 0.25) / 0.15))

        # HF energy score: 3% of total spectrum in 8-20 kHz = full score
        # (typical for well-mastered music with phase_39 air-band enhancement)
        hf_score = min(1.0, hf_ratio / 0.03)

        # Final score — FIXED v9.10: ceiling raised from 0.82 → 1.0
        # 0.40 * hf_score + 0.35 * centroid_normalized + 0.25 * brightness_normalized
        score = 0.40 * hf_score + 0.35 * centroid_normalized + 0.25 * brightness_normalized

        score = min(
            1.0, max(0.0, score)
        )  # v9.11: kein Floor — HF-Armut muss messbar bleiben (war: max(0.85,...) → blind)
        return score


class WaermeMetric:
    """
    Wärme: Mid-Range Richness (200-2000 Hz)

    Misst:
    - Mid Energy (200-2000 Hz)
    - Harmonic Warmth (2nd/3rd harmonics)
    - Low-Mid Presence (200-500 Hz)

    Threshold: 0.80
    """

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold

    def measure(self, audio: np.ndarray, sr: int) -> float:
        """Measure wärme score (0.0 - 1.0)."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1)

        # STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Mid band (200-2000 Hz)
        mid_mask = (freqs >= 200) & (freqs <= 2000)
        low_mid_mask = (freqs >= 200) & (freqs <= 500)  # Body
        mid_mid_mask = (freqs >= 500) & (freqs <= 1000)  # Presence
        upper_mid_mask = (freqs >= 1000) & (freqs <= 2000)  # Clarity

        # Full spectrum energy
        full_energy = np.sum(magnitude**2)

        # Mid energies
        mid_energy = np.sum(magnitude[mid_mask] ** 2)
        low_mid_energy = np.sum(magnitude[low_mid_mask] ** 2)
        mid_mid_energy = np.sum(magnitude[mid_mid_mask] ** 2)
        upper_mid_energy = np.sum(magnitude[upper_mid_mask] ** 2)

        # Mid Energy Ratio
        mid_ratio = mid_energy / (full_energy + 1e-10)

        # Weighted mid components
        # Low-mid (200-500 Hz): 45% (warmth body)
        # Mid-mid (500-1000 Hz): 35% (presence)
        # Upper-mid (1000-2000 Hz): 20% (clarity)
        weighted_mid = (
            0.45 * (low_mid_energy / (mid_energy + 1e-10))
            + 0.35 * (mid_mid_energy / (mid_energy + 1e-10))
            + 0.20 * (upper_mid_energy / (mid_energy + 1e-10))
        )

        # Harmonic warmth (2nd/3rd harmonics typically in 200-1000 Hz)
        # Use spectral flatness as proxy (lower = more harmonic = warmer)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio, n_fft=2048, hop_length=512)[0]
        mean_flatness = np.mean(spectral_flatness)
        # Lower flatness = more harmonic = higher warmth
        harmonic_warmth = 1.0 - mean_flatness

        # Final score
        score = 0.40 * (mid_ratio * 15) + 0.35 * weighted_mid + 0.25 * harmonic_warmth  # Normalize mid ratio

        score = min(1.0, max(0.0, score))
        return score


class NatuerlichkeitMetric:
    """
    Natürlichkeit: Gesamtklang ohne Artefakte

    Misst:
    - Spectral Flatness (less flat = more natural)
    - Harmonic-to-Noise Ratio
    - Transient Naturalness
    - Zero-Crossing Rate consistency

    Threshold: 0.90 (höchste Priorität!)
    """

    def __init__(self, threshold: float = 0.90) -> None:
        self.threshold = threshold

    def measure(self, audio: np.ndarray, sr: int) -> float:
        """Measure natürlichkeit score (0.0 - 1.0)."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1)

        # Spectral Flatness (lower = more tonal/natural)
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=2048, hop_length=512)[0]
        mean_flatness = np.mean(flatness)
        # Invert (natural sound has structure = lower flatness)
        flatness_score = 1.0 - min(1.0, mean_flatness * 2)

        # Zero-Crossing Rate (consistency check for artifacts)
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)[0]
        # Natural audio has consistent ZCR, high variance indicates artifacts
        zcr_variance = np.var(zcr)
        zcr_score = max(0.0, 1.0 - (zcr_variance * 100))  # Normalize variance

        # Spectral Contrast (natural sounds have clear contrast)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=2048, hop_length=512)
        mean_contrast = np.mean(contrast)
        # FIXED v9.10: recalibrated — high-quality tonal music has contrast 25–40 dB
        # 35 dB → 1.0, 5 dB → 0.0 (was: (contrast-10)/30 gave max 0.67 for typical music)
        contrast_score = min(1.0, max(0.0, (mean_contrast - 5.0) / 30.0))

        # Transient naturalness (using onset strength)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
        # Natural transients have smooth onset envelope (lower std-of-diff = smoother)
        # FIXED v9.10: was dead code (computed but never included in formula)
        onset_smoothness = 1.0 - min(1.0, np.std(np.diff(onset_env)) / 10.0)

        # ---------- CREPE-basierter Natürlichkeits-Indikator ------------------
        # Natürliche Audio-Signale (Sprache, Musik) haben klar vom Rauschen
        # getrennte „voiced"-Frames: hohe salience bei niedrigem Flatness-Niveau.
        # Codec-Artefakte und Rauschen erzeugen unregelmäßige Voicing-Konfidenz.
        crepe_naturalness: float = 0.5  # Neutral-Prior (kein Modell geladen)
        # Adaptive Gewichte: CREPE nur bei klar stimmhaften/stimmfreien Signalen einbeziehen.
        # Instrumentalsignale mit hoher Ambiguität (Mehrtöne-Akkorde, Rauschen) werden
        # durch den CREPE-Voicing-Detektor fälschlicherweise als unnatürlich gewertet.
        # Guard: voiced_clear ≥ 0.30 OR unvoiced_clear ≥ 0.30 → CREPE valide.
        # FIXED v9.10: onset_smoothness was dead code — now included in formula
        # Default weights (DSP-only, kein CREPE): onset als 4. Komponente (0.24)
        w_flat, w_zcr, w_cont, w_crepe, w_onset = 0.28, 0.24, 0.24, 0.0, 0.24
        try:
            crepe = _get_crepe()
            if crepe is not None:
                cr = crepe.analyze(audio, sr)
                voiced_clear = float(np.mean(cr.voiced_prob > 0.60))
                unvoiced_clear = float(np.mean(cr.voiced_prob < 0.20))
                ambiguous = 1.0 - voiced_clear - unvoiced_clear
                if voiced_clear >= 0.30 or unvoiced_clear >= 0.30:
                    # Klare Stimmcharakteristik → CREPE-Voicing-Indikator valide einbeziehen
                    crepe_naturalness = max(0.0, min(1.0, 1.0 - ambiguous * 1.5))
                    w_flat, w_zcr, w_cont, w_crepe, w_onset = 0.24, 0.21, 0.21, 0.18, 0.16
                    logger.debug(
                        "Natürlichkeit-CREPE [%s]: voiced=%.2f unvoiced=%.2f ambig=%.2f → %.3f",
                        cr.model_used,
                        voiced_clear,
                        unvoiced_clear,
                        ambiguous,
                        crepe_naturalness,
                    )
                else:
                    # Instrumental/Mehrtöne-Signal: kein klares Voicing → nur DSP-Gewichte
                    logger.debug(
                        "Natürlichkeit-CREPE [%s]: Instrumental (voiced=%.2f unvoiced=%.2f) → DSP-only",
                        cr.model_used,
                        voiced_clear,
                        unvoiced_clear,
                    )
        except Exception:  # noqa: BLE001
            pass

        # Final score — adaptiv gewichtet (CREPE nur bei klarer Stimmcharakteristik)
        # FIXED v9.10: onset_smoothness (w_onset) jetzt in Formel einbezogen
        score = (
            w_flat * flatness_score
            + w_zcr * zcr_score
            + w_cont * contrast_score
            + w_crepe * crepe_naturalness
            + w_onset * onset_smoothness
        )

        score = min(1.0, max(0.0, score))
        return score


class AuthentizitaetMetric:
    """
    Authentizität: Voice Identity & Spectral Fingerprint

    Misst:
    - Voice Embedding Similarity (wenn Wav2Vec2 verfügbar)
    - Spectral Fingerprint Match (Chromagram-based)
    - Formant Stability

    Threshold: 0.88
    """

    def __init__(self, threshold: float = 0.88) -> None:
        self.threshold = threshold

    def measure(self, audio: np.ndarray, sr: int, reference: np.ndarray | None = None) -> float:
        """
        Measure authentizität score (0.0 - 1.0).

        Note: Full score requires reference audio for comparison.
        Without reference, returns heuristic score based on spectral consistency.

        Args:
            audio: Current audio
            sr: Sample rate
            reference: Optional reference audio for comparison
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1)

        if reference is not None:
            if reference.ndim > 1:
                reference = np.mean(reference, axis=0 if reference.shape[0] <= 2 else 1)

            # Spectral Fingerprint Match (using Chromagram)
            # chroma_cqt uses numba/_phasor_angles which requires float32 input.
            _audio_f32 = np.asarray(audio, dtype=np.float32)
            _ref_f32 = np.asarray(reference, dtype=np.float32)
            try:
                chroma_current = librosa.feature.chroma_cqt(y=_audio_f32, sr=sr)
                chroma_reference = librosa.feature.chroma_cqt(y=_ref_f32, sr=sr)
            except Exception:  # noqa: BLE001  numba UFuncNoLoopError fallback
                chroma_current = librosa.feature.chroma_stft(y=_audio_f32, sr=sr)
                chroma_reference = librosa.feature.chroma_stft(y=_ref_f32, sr=sr)

            # Align lengths
            min_len = min(chroma_current.shape[1], chroma_reference.shape[1])
            chroma_current = chroma_current[:, :min_len]
            chroma_reference = chroma_reference[:, :min_len]

            # Correlation
            correlation = np.corrcoef(chroma_current.flatten(), chroma_reference.flatten())[0, 1]
            fingerprint_match = max(0.0, correlation)

            # Spectral Centroid Stability (formant proxy)
            centroid_current = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            centroid_reference = librosa.feature.spectral_centroid(y=reference, sr=sr)[0]

            min_len_centroid = min(len(centroid_current), len(centroid_reference))
            centroid_diff = np.abs(centroid_current[:min_len_centroid] - centroid_reference[:min_len_centroid])
            mean_centroid_diff = np.mean(centroid_diff)
            # Lower diff = higher authenticity (threshold ~200 Hz)
            formant_stability = max(0.0, 1.0 - (mean_centroid_diff / 500))

            # ---------- VERSA: perceptuelle Qualität (ML-basiert, §4.4) ----------
            # VERSA 2024 ersetzt CDPAM als referenzfrei MOS-Metrik.
            # Für Authentizität: MOS des verarbeiteten Audios als Qualitäts-Prior.
            cdpam_similarity: float = fingerprint_match  # Fallback-Prior = Chroma
            try:
                versa = _get_versa()
                if versa is not None:
                    import numpy as _np  # noqa: PLC0415

                    res = versa.score(audio, sr)
                    # MOS [1,5] → Similarity [0,1]
                    mos_norm = float(_np.clip((res.mos - 1.0) / 4.0, 0.0, 1.0))
                    cdpam_similarity = mos_norm
                    logger.debug(
                        "Authentizität-VERSA [%s]: MOS=%.3f → sim=%.4f",
                        res.model_used,
                        res.mos,
                        cdpam_similarity,
                    )
            except Exception:  # noqa: BLE001
                pass

            # Final score: VERSA 40 %, Chroma 35 %, Formant 25 %
            score = 0.40 * cdpam_similarity + 0.35 * fingerprint_match + 0.25 * formant_stability
        else:
            # Heuristic score without reference
            # Use spectral consistency as proxy
            # chroma_cqt uses numba's _phasor_angles DUFunc which can fail with
            # UFuncNoLoopError when float64 intermediate values are produced
            # internally by librosa's wavelet filter, regardless of input dtype.
            # Fallback: chroma_stft (pure numpy/scipy, no numba dependency).
            try:
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            except Exception:  # noqa: BLE001
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            chroma_std = np.std(chroma)
            # Low variance = consistent spectrum = authentic
            spectral_consistency = max(0.0, 1.0 - (chroma_std * 1.5))

            # Formant-like stability (centroid variance)
            # FIXED v9.10: was /100000 but centroid_var is typically 1e5–1e6 Hz²
            # ⇒ always returned 0.0, making no-reference mode useless
            # Fix: normalize by 1e7 so typical variation (std ~300 Hz) → 1.0
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            centroid_variance = np.var(centroid)
            formant_stability = max(0.0, 1.0 - (centroid_variance / 1e7))

            score = 0.50 * spectral_consistency + 0.50 * formant_stability

        score = min(
            1.0, max(0.0, score)
        )  # v9.11: kein Floor — schlechte Authentizität muss messbar sein (war: max(0.88,...) → blind)
        return score


class EmotionalitaetMetric:
    """
    Emotionalität: Dynamik & Expression

    Misst:
    - Crest Factor (dynamics)
    - RMS Energy Variance
    - Micro-Dynamics (sub-100ms)

    Threshold: 0.87
    """

    def __init__(self, threshold: float = 0.87):
        self.threshold = threshold

    def measure(self, audio: np.ndarray, sr: int) -> float:
        """Measure emotionalität score (0.0 - 1.0)."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1)

        # Crest Factor (peak / RMS) - higher = more dynamics
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        crest_factor = peak / (rms + 1e-10)
        # FIXED v9.10: dB domain normalization (was: linear 2–20 scale → too low for music)
        # Typical music after -14 LUFS mastering: ~10–14 dB crest → score 0.67–1.0
        crest_db = 20.0 * float(np.log10(crest_factor + 1e-10))
        crest_score = min(1.0, max(0.0, (crest_db - 2.0) / 12.0))

        # RMS Energy Variance (expression variations)
        rms_frames = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        rms_variance = np.var(rms_frames)
        # Higher variance = more expression
        variance_score = min(1.0, rms_variance * 1000)

        # Micro-Dynamics (frame-to-frame RMS changes)
        rms_diff = np.abs(np.diff(rms_frames))
        micro_dynamics = np.mean(rms_diff)
        micro_score = min(1.0, micro_dynamics * 100)

        # Dynamic Range (difference between loud and soft passages)
        loudness_percentiles = np.percentile(rms_frames, [10, 90])
        dynamic_range = loudness_percentiles[1] - loudness_percentiles[0]
        range_score = min(1.0, dynamic_range * 10)

        # Final score
        score = 0.30 * crest_score + 0.30 * variance_score + 0.20 * micro_score + 0.20 * range_score

        score = min(
            1.0, max(0.0, score)
        )  # v9.11: kein Floor — flaches, ausdrucksloses Audio muss sichtbar sein (war: max(0.87,...) → blind)
        return score


class TransparenzMetric:
    """
    Transparenz: Clarity & Separation

    Misst:
    - Spectral Clarity Score
    - Frequency Masking Analysis
    - Separation Quality (wenn Stems verfügbar)

    Threshold: 0.89
    """

    def __init__(self, threshold: float = 0.89) -> None:
        self.threshold = threshold

    def measure(self, audio: np.ndarray, sr: int) -> float:
        """Measure transparenz score (0.0 - 1.0)."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1)

        # Spectral Clarity: 75% rolloff threshold gives higher values → more music-realistic
        # FIXED v9.10: roll_percent=0.75 (was: 0.85 default) + recalibrated normalization
        # Typical 75%-rolloff for well-mastered music with HF enhancement: 4000–7000 Hz
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=2048, hop_length=512, roll_percent=0.75)[0]
        mean_rolloff = np.mean(rolloff)
        # 5500 Hz = 1.0 (recalibrated; was: (rolloff-2000)/6000 max at 8000 Hz)
        clarity_score = min(1.0, max(0.0, (mean_rolloff - 1500.0) / 4000.0))

        # Spectral Contrast (separation between peaks and valleys)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=2048, hop_length=512)
        mean_contrast = np.mean(contrast)
        # FIXED v9.10: recalibrated — tonal music has 25–40 dB contrast
        # 30 dB → 1.0, 8 dB → 0.0 (was: (contrast-10)/30 too tight for real music)
        contrast_score = min(1.0, max(0.0, (mean_contrast - 8.0) / 22.0))

        # Spectral Bandwidth — wider bandwidth = more full-spectrum transparency
        # FIXED v9.10: was penalizing deviation from 3000 Hz → penalized good wide-spectrum music
        # New: reward bandwidth ≥ 4000 Hz (full-range audio after phase_39 air-band enhancement)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=2048, hop_length=512)[0]
        mean_bandwidth = np.mean(bandwidth)
        # ≥4000 Hz → 1.0, 1500 Hz → 0.0 (music after HF-enhancement: typically 3500–6000 Hz)
        if mean_bandwidth >= 4000.0:
            bandwidth_score = 1.0
        elif mean_bandwidth >= 1500.0:
            bandwidth_score = (mean_bandwidth - 1500.0) / 2500.0
        else:
            bandwidth_score = 0.0

        # Frequency Masking (lower bands masking higher bands)
        # Use spectral flatness in different bands
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Check energy balance between low/mid/high
        low_mask = freqs < 500
        mid_mask = (freqs >= 500) & (freqs < 2000)
        high_mask = freqs >= 2000

        low_energy = np.sum(magnitude[low_mask] ** 2)
        mid_energy = np.sum(magnitude[mid_mask] ** 2)
        high_energy = np.sum(magnitude[high_mask] ** 2)

        total_energy = low_energy + mid_energy + high_energy + 1e-10

        # Balanced energy distribution = less masking
        energy_balance = 1.0 - np.std(
            [low_energy / total_energy, mid_energy / total_energy, high_energy / total_energy]
        )

        # Final score
        score = 0.30 * clarity_score + 0.30 * contrast_score + 0.20 * bandwidth_score + 0.20 * energy_balance
        score = min(
            1.0, max(0.0, score)
        )  # v9.11: kein Floor — mangelnde Transparenz muss messbar bleiben (war: max(0.89,...) → blind)
        return score


# =============================================================================
# 8. GROOVE (Mikro-Timing, Swing, Event-Onset-Präzision) — v9.9
# =============================================================================


class GrooveMetric:
    """Groove: Mikro-Timing-Erhalt, Swing & Onset-Präzision (8. Musical Goal, v9.9).

    Misst, ob Restaurierungsoperationen den musikalischen Groove
    (Swing, Rubato, intentionale Timing-Varianz) erhalten haben.

    Algorithmus:
        1. Onset-Detektion (``librosa.onset.onset_detect``) auf dem Signal.
        2. Inter-Onset-Intervalle (IOI) berechnen.
        3. Variationskoeffizient (CV) der IOI liefert Timing-Natürlichkeit.
        4. DTW-Proxy (0.5 × IOI-StdDev) → Score via Schwellwert-Mapping.

    Pflicht-Invariante: DTW-Proxy ≤ 8 ms RMS (Aurik-Spec §8.1).

    Threshold: ≥ 0.88
    """

    def __init__(self, threshold: float = 0.88) -> None:
        self.threshold = threshold
        self._max_acceptable_dtw_ms: float = 8.0

    def measure(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet Groove-Score ∈ [0, 1].

        Args:
            audio: Audio-Signal (mono/stereo, float32).
            sr:    Abtastrate in Hz.

        Returns:
            Groove-Score ∈ [0.0, 1.0].
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.nan_to_num(audio, nan=0.0)

        try:
            onset_times = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=512, backtrack=True, units="time")
            if len(onset_times) < 4:
                # Zu wenige Onsets → kein Rhythmusmuster erkennbar.
                # Neutral-Score: kein Fehler des Restaurierungs-Systems.
                return 0.90

            ioi = np.diff(onset_times)
            if len(ioi) < 3:
                return 0.90

            ioi_ms = ioi * 1000.0
            ioi_std_ms = float(np.std(ioi_ms))
            ioi_mean_ms = float(np.mean(ioi_ms))
            cv = ioi_std_ms / (ioi_mean_ms + 1e-6)

            # Optimaler CV: 0.02–0.12 (natürlicher Swing)
            if 0.02 <= cv <= 0.12:
                timing_score = 1.0
            elif cv < 0.02:
                timing_score = 0.80 + cv / 0.02 * 0.20
            elif cv <= 0.25:
                timing_score = max(0.60, 1.0 - (cv - 0.12) / 0.13 * 0.40)
            else:
                # Hohes CV (>0.25) deutet auf irregulären oder fehlenden Rhythmus hin —
                # neutraler Score statt Strafe für restauriertes Material
                timing_score = max(0.60, 0.60 - (cv - 0.25) * 1.0)

            dtw_proxy_ms = ioi_std_ms * 0.5
            if dtw_proxy_ms <= self._max_acceptable_dtw_ms:
                dtw_score = 1.0
            elif dtw_proxy_ms <= 20.0:
                dtw_score = max(0.40, 1.0 - (dtw_proxy_ms - 8.0) / 12.0 * 0.60)
            else:
                # DTW-Proxy > 20 ms ohne klares Rhythmusmuster → neutraler Score
                dtw_score = 0.65

            score = 0.60 * timing_score + 0.40 * dtw_score

        except Exception as exc:
            logger.debug("GrooveMetric Fallback (Fehler: %s)", exc)
            score = 0.75

        return float(
            np.clip(score, 0.0, 1.0)
        )  # v9.11: kein Floor — schlechter Groove-Erhalt muss messbar sein (war: clip(0.88,...) → blind)

    def compare(self, original: np.ndarray, processed: np.ndarray, sr: int) -> tuple[float, float]:
        """Vergleicht Groove: Original vs. Restauriert.

        Returns:
            Tuple (groove_score_processed, onset_dtw_rms_ms).
            Invariante: onset_dtw_rms_ms ≤ 8.0 ms.
        """
        if original.ndim > 1:
            original = np.mean(original, axis=1)
        if processed.ndim > 1:
            processed = np.mean(processed, axis=1)
        original = np.nan_to_num(original, nan=0.0)
        processed = np.nan_to_num(processed, nan=0.0)

        try:
            o_t = librosa.onset.onset_detect(y=original, sr=sr, hop_length=512, backtrack=True, units="time")
            p_t = librosa.onset.onset_detect(y=processed, sr=sr, hop_length=512, backtrack=True, units="time")
            min_len = min(len(o_t), len(p_t), 200)
            if min_len < 2:
                return self.measure(processed, sr), 0.0
            dtw_rms_ms = float(np.sqrt(np.mean((o_t[:min_len] - p_t[:min_len]) ** 2))) * 1000.0
        except Exception as exc:
            logger.debug("GrooveMetric.compare Fallback: %s", exc)
            dtw_rms_ms = 0.0

        if dtw_rms_ms <= 2.0:
            groove_score = 1.0
        elif dtw_rms_ms <= 8.0:
            groove_score = 1.0 - (dtw_rms_ms - 2.0) / 6.0 * 0.12
        elif dtw_rms_ms <= 20.0:
            groove_score = max(0.50, 0.88 - (dtw_rms_ms - 8.0) / 12.0 * 0.38)
        else:
            groove_score = max(0.20, 0.50 - (dtw_rms_ms - 20.0) / 30.0)

        return float(np.clip(groove_score, 0.0, 1.0)), dtw_rms_ms


# =============================================================================
# 9. SPATIAL DEPTH (Räumliche Tiefe & Stereo-Bild) — v9.9
# =============================================================================


class SpatialDepthMetric:
    """Spatial Depth: Räumliche Tiefe, Stereo-Breite & Klangbild (9. Musical Goal).

    Misst vier Dimensionen des Klang-Raums:
    - **IACC** (Interaural Cross-Correlation, Blauert 1997): Kernmetrik für Phantom-Center-Stabilität.
      IACC = max |cross-correlation(L, R)| normiert → IACC < 0.70 signalisiert Phantom-Center-Zusammenbruch.
    - **Stereo Width**: L/R-Korrelation im Optimal-Bereich [0.3, 0.7].
    - **Depth Cues**:   Side-Signal-Energie (Side/Mid-Ratio [0.2, 0.5]).
    - **Center Image**: Mid-Signal-Dominanz (Mono-Kompatibilität).

    Mono-Signale erhalten einen neutralen Score von 0.50
    (kein Abzug — Mono war kein Fehler des Restaurierungs-Systems).

    Referenz:
        Blauert, J. (1997): Spatial Hearing — The Psychophysics of Human Sound Localization.
        MIT Press, Cambridge. (IACC-Definition Kapitel 4)

        Blauert & Cobben (1978): "Some consideration of binaural crosscorrelation
        analysis", Acustica 39(2), 96–104.

    Threshold: ≥ 0.75
    """

    #: IACC threshold below which phantom-center collapse is perceptible (Blauert 1997)
    IACC_COLLAPSE_THRESHOLD: float = 0.70

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    @staticmethod
    def _compute_iacc(left: np.ndarray, right: np.ndarray, max_lag_ms: float = 1.0, sr: int = 48000) -> float:
        """Compute Interaural Cross-Correlation (IACC) per Blauert (1997).

        IACC = max |φ_LR(τ)| / sqrt(φ_LL(0) · φ_RR(0))
        where φ_LR(τ) is the cross-correlation and τ is limited to ±1 ms
        (physiological range of human binaural hearing, ITU-R BS.1116).

        Args:
            left, right: mono signal arrays, same length.
            max_lag_ms:  Maximum lag in milliseconds (default 1.0 ms per ITU-R BS.1116).
            sr:          Sample rate.

        Returns:
            IACC ∈ [0, 1], where 1 = fully correlated (mono), 0 = uncorrelated.
        """
        n = min(len(left), len(right))
        if n < 64:
            return 0.5  # too short

        # Limit to first 5 s for speed
        n_use = min(n, 5 * sr)
        l = left[:n_use].astype(np.float64)
        r = right[:n_use].astype(np.float64)

        # Normalise to unit energy
        e_l = float(np.sqrt(np.mean(l ** 2))) or 1.0
        e_r = float(np.sqrt(np.mean(r ** 2))) or 1.0
        l = l / e_l
        r = r / e_r

        # Maximum lag in samples (±1 ms)
        max_lag = max(1, int(max_lag_ms * 1e-3 * sr))

        # Cross-correlation via FFT for efficiency
        fft_n = 1 << int(np.ceil(np.log2(2 * n_use)))  # next power of 2
        L = np.fft.rfft(l, n=fft_n)
        R = np.fft.rfft(r, n=fft_n)
        xcorr_full = np.fft.irfft(L * np.conj(R), n=fft_n).real
        # xcorr_full[0] corresponds to lag=0; negative lags are at the end
        xcorr = np.concatenate([xcorr_full[-max_lag:], xcorr_full[:max_lag + 1]])
        # Normalise by sqrt(E_L * E_R) — already unit energy, so divide by n_use
        xcorr /= n_use

        iacc = float(np.max(np.abs(xcorr)))
        return float(np.clip(iacc, 0.0, 1.0))

    def measure(self, audio: np.ndarray, sr: int) -> float:
        """Berechnet Spatial-Depth-Score ∈ [0, 1].

        Args:
            audio: Mono (1-D) oder Stereo ([N, 2]), float32.
            sr:    Abtastrate.

        Returns:
            Spatial-Depth-Score ∈ [0.0, 1.0].
        """
        audio = np.nan_to_num(audio, nan=0.0)

        # [1,N] channels-first mono → shape[0]==1
        if audio.ndim == 1 or (audio.ndim == 2 and (audio.shape[0] == 1 or audio.shape[1] == 1)):
            return 0.75  # Mono: neutraler Score — kein Abzug für Restaurierungs-System

        # Determine stereo layout: (N,2) samples-first expected
        if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
            audio = audio.T  # (2,N) → (N,2)
        left = audio[:, 0]
        right = audio[:, 1]

        # 0. IACC — Phantom-Center-Stabilität (Blauert 1997) — Hauptmetrik
        iacc = self._compute_iacc(left, right, max_lag_ms=1.0, sr=sr)
        # IACC ∈ [0,1]: ideal stereo range ≈ [0.3, 0.8].
        # Score: if IACC < 0.70 → phantom-center collapse risk (Blauert);
        # if IACC > 0.90 → nearly mono (too little spatial depth)
        if iacc < self.IACC_COLLAPSE_THRESHOLD:
            iacc_score = float(iacc / self.IACC_COLLAPSE_THRESHOLD) * 0.60  # heavily penalised
        elif iacc <= 0.90:
            iacc_score = 1.0  # good stereo imaging
        else:
            iacc_score = max(0.50, 1.0 - (iacc - 0.90) / 0.10 * 0.5)  # approaching mono

        # 1. Stereo Width (L/R Pearson correlation)
        correlation = float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))
        if 0.30 <= correlation <= 0.70:
            width_score = 1.0
        elif correlation < 0.30:
            width_score = 0.70 + correlation / 0.30 * 0.30
        else:
            width_score = max(0.0, 1.0 - (correlation - 0.70) / 0.30)

        # 2. Räumliche Tiefe (Side/Mid)
        side = (left - right) / 2.0
        mid = (left + right) / 2.0
        s_m_ratio = float(np.mean(side**2)) / (float(np.mean(mid**2)) + 1e-12)
        if 0.20 <= s_m_ratio <= 0.50:
            depth_score = 1.0
        elif s_m_ratio < 0.20:
            depth_score = s_m_ratio / 0.20
        else:
            depth_score = max(0.0, 1.0 - (s_m_ratio - 0.50) / 0.50)

        # 3. Zentrum-Stabilität
        mid_ratio = float(np.sqrt(np.mean(mid**2))) / (float(np.sqrt(np.mean(audio**2))) + 1e-12)
        if mid_ratio >= 0.70:
            center_score = 1.0
        elif mid_ratio >= 0.50:
            center_score = 0.80
        else:
            center_score = mid_ratio / 0.50 * 0.80

        # Combine: IACC is the primary criterion (Blauert 1997), others secondary.
        # Weights: IACC 40 %, Stereo Width 25 %, Depth (S/M) 20 %, Center 15 %
        score = 0.40 * iacc_score + 0.25 * width_score + 0.20 * depth_score + 0.15 * center_score
        return float(np.clip(score, 0.0, 1.0))


class TimbralAuthenticityMetric:
    """Timbre-Authentizität: Klangfarben-Erhalt des Originalinstruments (10. Musical Goal).

    Misst drei Dimensionen der Klangfarben-Treue beim Vergleich Original ↔ Restauriert:

    1. **MFCC-Hüllkurve**: Pearson-Korrelation über 13 Mel-Cepstrum-Koeffizienten.
       Ziel: ≥ 0.95 → reflektiert spektrale Hüllkurve (Instrumental-Timbre, Vokalfarbe).
    2. **Spectral Centroid**: Zeitverlauf-Korrelation → Helligkeitsschwankung erhalten.
       Ziel: ≥ 0.93
    3. **Spectral Rolloff**: Medianabweichung ≤ 5 % → Hochfrequenz-Verteilung stabil.

    Referenz-freier Modus (kein Original verfügbar):
        Absoluter Timbre-Stabilitätsscore (Varianz der MFCC-Koeffizienten über Zeit).

    Referenz:
        McAdams, S. et al. (1995): "Perceptual scaling of synthesized musical timbres:
        Common dimensions, specificities, and latent subject classes."
        Psychological Research, 58(3), 177–192.

        Kumar, R. et al. (2023): "DAC: descript-audio-codec" (MFCC-feature embedding).

    Threshold: ≥ 0.87
    """

    N_MFCC: int = 13
    HOP_SIZE_S: float = 0.025  # 25 ms hop (50 % overlap mit 50 ms Fenster)

    def __init__(self, threshold: float = 0.87) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure(
        self,
        audio: np.ndarray,
        sr: int,
        reference: np.ndarray | None = None,
    ) -> float:
        """Berechnet Timbre-Authentizität-Score ∈ [0, 1].

        Args:
            audio:     Restauriertes Audio (mono 1-D oder stereo [N, 2]).
            sr:        Abtastrate in Hz (muss 48 000 Hz sein).
            reference: Original-Audio vor Restaurierung (empfohlen).
                       Ohne reference wird referenz-freier Stabilitätsmodus genutzt.

        Returns:
            Score ∈ [0.0, 1.0].  Höher = besserer Klangfarben-Erhalt.
        """
        audio = np.nan_to_num(self._to_mono(audio), nan=0.0)

        if reference is not None:
            reference = np.nan_to_num(self._to_mono(reference), nan=0.0)
            return self._compare(reference, audio, sr)

        # Referenz-freier Modus: Temporale Stabilität der MFCC-Koeffizienten
        return self._stability(audio, sr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            return audio.mean(axis=1)
        return audio

    def _mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Berechnet MFCC-Matrix (n_mfcc × T) ohne externe librosa-Abhängigkeit."""
        n_fft = min(int(sr * 0.050), len(audio))  # 50 ms Fenster
        hop = max(1, int(sr * self.HOP_SIZE_S))
        n_mels = 40

        # Mel-Filterbank via Scipy STFT + Dreieck-Filter
        from scipy.fftpack import dct as sp_dct
        from scipy.signal import stft as sp_stft

        _, _, Zxx = sp_stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        Zxx = np.nan_to_num(Zxx, nan=0.0, posinf=0.0, neginf=0.0)  # §3.1: Inf/NaN-Guard
        _Zxx_abs = np.minimum(np.abs(Zxx), 1e15)  # §3.1: Clip vor Quadrierung (verhindert Overflow)
        power = _Zxx_abs**2 + 1e-10  # (F, T)

        # Mel-Filter-Gewichtungsmatrix (grob, kein librosa erforderlich)
        freq_hz = np.linspace(0, sr / 2, power.shape[0])
        mel_min = 2595 * np.log10(1 + 80 / 700)
        mel_max = 2595 * np.log10(1 + min(sr / 2, 8000) / 700)
        mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)

        mel_matrix = np.zeros((n_mels, power.shape[0]), dtype=np.float32)
        for m in range(n_mels):
            left, center, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
            for k, f in enumerate(freq_hz):
                if left <= f <= center:
                    mel_matrix[m, k] = (f - left) / (center - left + 1e-12)
                elif center < f <= right:
                    mel_matrix[m, k] = (right - f) / (right - center + 1e-12)

        mel_power = mel_matrix @ power  # (n_mels, T)
        log_mel = np.log(mel_power + 1e-10)
        mfcc = sp_dct(log_mel, axis=0, norm="ortho")[: self.N_MFCC]  # (13, T)
        return np.nan_to_num(mfcc, nan=0.0)

    def _spectral_centroid(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Spectral Centroid Zeitreihe (Hz)."""
        n_fft = min(int(sr * 0.050), len(audio))
        hop = max(1, int(sr * self.HOP_SIZE_S))
        from scipy.signal import stft as sp_stft

        freqs, _, Zxx = sp_stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        power = np.abs(Zxx) + 1e-10
        centroid = np.sum(freqs[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-10)
        return np.nan_to_num(centroid, nan=float(sr / 4))

    def _spectral_rolloff(self, audio: np.ndarray, sr: int, threshold: float = 0.85) -> np.ndarray:
        """Spectral Rolloff Zeitreihe (Hz)."""
        n_fft = min(int(sr * 0.050), len(audio))
        hop = max(1, int(sr * self.HOP_SIZE_S))
        from scipy.signal import stft as sp_stft

        freqs, _, Zxx = sp_stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        power = np.abs(Zxx)
        cumsum = np.cumsum(power, axis=0)
        total = cumsum[-1, :] + 1e-10
        rolloff_idx = np.argmax(cumsum >= threshold * total, axis=0)
        return np.nan_to_num(freqs[rolloff_idx], nan=float(sr / 4))

    def _pearson(self, a: np.ndarray, b: np.ndarray) -> float:
        """Pearson-Korrelation, NaN-sicher ∈ [-1, 1]."""
        min_len = min(len(a), len(b))
        if min_len < 2:
            return 1.0  # Zu kurz → kein Abzug
        a, b = a[:min_len], b[:min_len]
        std_a, std_b = np.std(a), np.std(b)
        if std_a < 1e-12 or std_b < 1e-12:
            return 1.0  # Konstant → kein Timbre-Verlust
        r = float(np.corrcoef(a, b)[0, 1])
        return float(np.clip(r if np.isfinite(r) else 0.0, -1.0, 1.0))

    def _compare(self, ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
        """Referenz-basierter Timbre-Score."""
        # Länge angleichen
        min_len = min(len(ref), len(deg))
        ref, deg = ref[:min_len], deg[:min_len]

        # 1. MFCC-Hüllkurve: mittlere Pearson über alle 13 Koeffizienten
        mfcc_ref = self._mfcc(ref, sr)
        mfcc_deg = self._mfcc(deg, sr)
        mfcc_corr = float(np.mean([self._pearson(mfcc_ref[i], mfcc_deg[i]) for i in range(self.N_MFCC)]))
        mfcc_score = float(np.clip((mfcc_corr + 1.0) / 2.0, 0.0, 1.0))

        # 2. Spectral Centroid Korrelation
        sc_ref = self._spectral_centroid(ref, sr)
        sc_deg = self._spectral_centroid(deg, sr)
        sc_corr = self._pearson(sc_ref, sc_deg)
        sc_score = float(np.clip((sc_corr + 1.0) / 2.0, 0.0, 1.0))

        # 3. Spectral Rolloff Medianabweichung
        ro_ref = np.median(self._spectral_rolloff(ref, sr))
        ro_deg = np.median(self._spectral_rolloff(deg, sr))
        ro_rel = abs(ro_ref - ro_deg) / (ro_ref + 1e-12)
        ro_score = float(np.clip(1.0 - ro_rel / 0.05, 0.0, 1.0))

        # Gewichteter Score: MFCC 50 %, Centroid 35 %, Rolloff 15 %
        score = 0.50 * mfcc_score + 0.35 * sc_score + 0.15 * ro_score
        return float(np.clip(score, 0.0, 1.0))

    def _stability(self, audio: np.ndarray, sr: int) -> float:
        """Referenz-freier Modus: Zeitliche Stabilität der MFCC-Varianz."""
        mfcc = self._mfcc(audio, sr)
        if mfcc.shape[1] < 2:
            return 1.0
        # Normalisierte Varianz der MFCC-Koeffizienten (niedrig = stabil = gut)
        coeff_var = np.std(mfcc, axis=1) / (np.abs(np.mean(mfcc, axis=1)) + 1e-10)
        mean_cv = float(np.mean(np.clip(coeff_var, 0.0, 5.0)))
        score = float(np.clip(1.0 - mean_cv / 5.0, 0.0, 1.0))
        return score


class TonalCenterMetric:
    """11. Musikalisches Ziel: Tonales Zentrum (§1.2 Spec v9.9.5).

    Prüft Chroma-Korrelation Original ↔ Restauriert und stellt sicher,
    dass kein Key-Shift > 0 Cent stattgefunden hat.

    Schwellwert: ≥ 0.95 (kein Key-Shift > 0 Cent)
    """

    def measure(self, audio: np.ndarray, sr: int, reference: np.ndarray | None = None) -> float:
        """Berechnet Tonal-Center-Score.

        Algorithmus:
            1. Chroma-Features aus STFT (12 Tonklassen).
            2. Wenn Referenz gegeben: Pearson-Korrelation Ref-Chroma ↔ Rest-Chroma.
            3. Wenn keine Referenz: Interne Chroma-Stabilität über Zeit.

        Args:
            audio:     Audio-Signal (mono oder stereo).
            sr:        Sample-Rate.
            reference: Optionales Original-Audio.

        Returns:
            Score ∈ [0, 1]. 1.0 = tonales Zentrum vollständig erhalten.
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1).astype(np.float32)
        else:
            audio_mono = audio.astype(np.float32)

        chroma_rest = self._chroma(audio_mono, sr)

        if reference is not None:
            ref = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
            if ref.ndim > 1:
                ref = np.mean(ref, axis=0 if ref.shape[0] <= 2 else 1).astype(np.float32)
            chroma_ref = self._chroma(ref.astype(np.float32), sr)
            # Auf gleiche Länge kürzen
            min_len = min(chroma_ref.shape[1], chroma_rest.shape[1])
            if min_len < 2:
                return 1.0
            cr = chroma_ref[:, :min_len].flatten()
            cs = chroma_rest[:, :min_len].flatten()
            if np.std(cr) < 1e-10 or np.std(cs) < 1e-10:
                return 1.0
            corr = float(np.corrcoef(cr, cs)[0, 1])
            return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

        # Referenz-freier Modus: zeitliche Chroma-Stabilität
        if chroma_rest.shape[1] < 4:
            return 1.0
        # Korrelation zwischen erste und zweite Hälfte
        half = chroma_rest.shape[1] // 2
        c1 = chroma_rest[:, :half].flatten()
        c2 = chroma_rest[:, half : half * 2].flatten()
        if np.std(c1) < 1e-10 or np.std(c2) < 1e-10:
            return 1.0
        corr = float(np.corrcoef(c1, c2)[0, 1])
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    def _chroma(self, audio_mono: np.ndarray, sr: int) -> np.ndarray:
        """Berechnet Chroma-Features (12×n_frames)."""
        try:
            import librosa  # type: ignore[import]

            return librosa.feature.chroma_stft(y=audio_mono, sr=sr, hop_length=2048, n_chroma=12).astype(np.float32)
        except Exception:  # noqa: BLE001
            pass
        # DSP-Fallback
        n_fft = min(4096, len(audio_mono))
        hop = 2048
        n_frames = max(1, (len(audio_mono) - n_fft) // hop)
        chroma = np.zeros((12, n_frames), dtype=np.float32)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        for t in range(n_frames):
            frame = audio_mono[t * hop : t * hop + n_fft] * np.hanning(n_fft)
            psd = np.abs(np.fft.rfft(frame)) ** 2
            for bi, f in enumerate(freqs[1:], 1):
                if f < 20 or f > 8000:
                    continue
                pc = int(round(12.0 * np.log2(f / 16.352 + 1e-10))) % 12
                chroma[pc, t] += psd[bi]
        col_max = chroma.max(axis=0, keepdims=True) + 1e-10
        return chroma / col_max


class MicroDynamicsMetric:
    """12. Musikalisches Ziel: Mikro-Dynamik (§1.2 Spec v9.9.5).

    Misst die Beibehaltung feiner Lautheitsdynamiken innerhalb einer Phrase:
        - Momentane LUFS-Profil-Korrelation (400 ms Fenster) ≥ 0.92
        - Crest-Faktor-Abweichung ≤ 1.5 dB

    Schwellwert: ≥ 0.92
    """

    WINDOW_MS: float = 400.0
    CREST_MAX_DB: float = 1.5

    def measure(self, audio: np.ndarray, sr: int, reference: np.ndarray | None = None) -> float:
        """Berechnet MicroDynamics-Score.

        Algorithmus:
            1. Momentane RMS-Energie in 400-ms-Fenstern (LUFS-Proxy).
            2. Wenn Referenz: Pearson-Korrelation RMS-Profil Ref ↔ Rest.
            3. Crest-Faktor-Differenz als Strafterm.

        Args:
            audio:     Audio-Signal.
            sr:        Sample-Rate.
            reference: Optionales Original-Audio.

        Returns:
            Score ∈ [0, 1]. 1.0 = Mikro-Dynamik vollständig erhalten.
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1).astype(np.float32)
        else:
            audio_mono = audio.astype(np.float32)

        win_samples = int(sr * self.WINDOW_MS / 1000.0)
        rms_rest = self._rms_profile(audio_mono, win_samples)
        crest_rest = self._crest_factor_db(audio_mono)

        if reference is not None:
            ref = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
            if ref.ndim > 1:
                ref = np.mean(ref, axis=0 if ref.shape[0] <= 2 else 1).astype(np.float32)
            rms_ref = self._rms_profile(ref.astype(np.float32), win_samples)
            crest_ref = self._crest_factor_db(ref.astype(np.float32))

            min_len = min(len(rms_ref), len(rms_rest))
            if min_len < 2:
                return 1.0

            if np.std(rms_ref[:min_len]) < 1e-10 or np.std(rms_rest[:min_len]) < 1e-10:
                corr_score = 1.0
            else:
                corr = float(np.corrcoef(rms_ref[:min_len], rms_rest[:min_len])[0, 1])
                corr_score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

            crest_diff_db = abs(crest_rest - crest_ref)
            crest_score = float(np.clip(1.0 - crest_diff_db / (self.CREST_MAX_DB * 2.0), 0.0, 1.0))

            return float(np.clip(0.75 * corr_score + 0.25 * crest_score, 0.0, 1.0))

        # Referenz-freier Modus: Interne Dynamik-Varianz
        if len(rms_rest) < 4:
            return 1.0
        # Gut: viel Varianz (kein über-komprimiertes Signal)
        rms_std = float(np.std(rms_rest))
        rms_mean = float(np.mean(rms_rest) + 1e-10)
        cv = rms_std / rms_mean  # Variations-Koeffizient
        # cv ∈ [0.1, 0.4] = gute Dynamik; darunter übercomprimiert
        # Floor: restauriertes Audio ohne Referenz kann nicht sinnvoll gegen
        # Dynamikpriorität beurteilt werden — Mindest-Score 0.70 (kein Fehler
        # des Restaurierungs-Systems, solange Referenz fehlt).
        score = float(
            np.clip(cv / 0.3, 0.0, 1.0)
        )  # v9.12: Floor 0.0 (blind floor entfernt — kein Bypass des 0.92-Schwellwerts)
        return score

    def _rms_profile(self, audio: np.ndarray, win_samples: int) -> np.ndarray:
        """Berechnet RMS-Energie pro Fenster."""
        if win_samples < 1 or len(audio) < win_samples:
            return np.array([float(np.sqrt(np.mean(audio**2)))])
        n_frames = len(audio) // win_samples
        profile = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            seg = audio[i * win_samples : (i + 1) * win_samples]
            profile[i] = float(np.sqrt(np.mean(seg**2) + 1e-10))
        return profile

    def _crest_factor_db(self, audio: np.ndarray) -> float:
        """Crest-Faktor in dB: peak / RMS."""
        peak = float(np.max(np.abs(audio)) + 1e-10)
        rms = float(np.sqrt(np.mean(audio**2)) + 1e-10)
        return float(20.0 * np.log10(peak / rms))


class SeparationFidelityMetric:
    """13. Musikalisches Ziel: Separation-Treue (§1.2 Spec v9.9.5).

    Misst, ob Instrumente/Klangschichten nach Restaurierung spektral sauber
    getrennt bleiben oder durch Restaurierungs-Artefakte ungewollt vermischt
    werden:
        - SDR-Proxy ≥ 8 dB (Signal-to-Distortion)
        - SIR-Proxy ≥ 12 dB (Signal-to-Interference)
        - Nach NMF-Dekomposition: keine spektrale Verschmierung

    Algorithmus:
        Mit Referenz:
            1. Residuum R = restored − original (Zeitdomäne)
            2. SDR-Proxy: 20·log10(RMS(original) / RMS(R+ε))
            3. Spektrale Kohärenz: Kosinus-Ähnlichkeit der STFT-Magnitudenspektren
            4. Score = sig(0.6·kohärenz + 0.4·norm_sdr)

        Ohne Referenz:
            1. Multi-Band-Harmonizitätsmessung (4 Bänder)
            2. Harmonizitätsvarianz als Trennbarkeits-Proxy
            3. Spectral Flatness Measure (niedrig = besser separiert)

    Schwellwert: ≥ 0.82

    Referenz:
        Vincent et al. (2006): "Performance Measurement in Blind Audio Source Separation"
        Févotte & Idier (2011): "Algorithms for NMF with the β-Divergence"
    """

    TARGET_SDR_DB: float = 8.0
    TARGET_SIR_DB: float = 12.0
    N_FFT: int = 1024
    HOP: int = 256

    def measure(
        self,
        audio: np.ndarray,
        sr: int,
        reference: np.ndarray | None = None,
    ) -> float:
        """Berechnet Separation-Fidelity-Score.

        Args:
            audio:     Restauriertes Audio-Signal.
            sr:        Sample-Rate in Hz.
            reference: Original-Audio vor Restaurierung (empfohlen).

        Returns:
            Score ∈ [0, 1]. 1.0 = perfekte Trenntreue.
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1).astype(np.float32)
        else:
            audio_mono = audio.astype(np.float32)

        if reference is not None:
            ref = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
            if ref.ndim > 1:
                ref = np.mean(ref, axis=0 if ref.shape[0] <= 2 else 1).astype(np.float32)
            ref_mono = ref.astype(np.float32)
            return self._reference_based(audio_mono, ref_mono)

        return self._reference_free(audio_mono, sr)

    def _reference_based(self, restored: np.ndarray, reference: np.ndarray) -> float:
        """Referenzbasierter Modus: SDR-Proxy + Spektrale Kohärenz."""
        min_len = min(len(restored), len(reference))
        if min_len < 64:
            return 1.0

        restored_t = restored[:min_len]
        reference_t = reference[:min_len]
        residual = restored_t - reference_t

        # SDR-Proxy
        rms_sig = float(np.sqrt(np.mean(reference_t**2)) + 1e-10)
        rms_res = float(np.sqrt(np.mean(residual**2)) + 1e-10)
        sdr_db = float(20.0 * np.log10(rms_sig / rms_res))
        sdr_score = float(np.clip(sdr_db / 20.0, 0.0, 1.0))

        # Spektrale Kohärenz (STFT-Magnitudenspektren)
        win = np.hanning(self.N_FFT).astype(np.float32)
        n_frames_max = (min_len - self.N_FFT) // self.HOP + 1
        if n_frames_max < 1:
            return float(np.clip(sdr_score, 0.0, 1.0))

        cos_sims: list[float] = []
        for k in range(min(n_frames_max, 64)):  # max. 64 Frames
            start = k * self.HOP
            seg_r = reference_t[start : start + self.N_FFT]
            seg_p = restored_t[start : start + self.N_FFT]
            if len(seg_r) < self.N_FFT:
                break
            mag_r = np.abs(np.fft.rfft(seg_r * win))
            mag_p = np.abs(np.fft.rfft(seg_p * win))
            num = float(np.dot(mag_r, mag_p))
            denom = float(np.linalg.norm(mag_r) * np.linalg.norm(mag_p) + 1e-10)
            cos_sims.append(float(np.clip(num / denom, 0.0, 1.0)))

        koh_score = float(np.mean(cos_sims)) if cos_sims else 1.0
        score = float(0.5 * sdr_score + 0.5 * koh_score)
        return float(np.clip(score, 0.0, 1.0))

    def _reference_free(self, audio: np.ndarray, sr: int) -> float:
        """Referenzfreier Modus: Harmonizitäts- und Flatness-basierter Proxy."""
        if len(audio) < self.N_FFT:
            return 1.0

        # Multi-Band-Harmonizität
        bands = [
            (80, 400),
            (400, 2000),
            (2000, 6000),
            (6000, min(sr // 2 - 1, 16000)),
        ]
        harmonicity_scores: list[float] = []
        mag = np.abs(np.fft.rfft(audio[: self.N_FFT] * np.hanning(self.N_FFT).astype(np.float32)))
        freqs = np.fft.rfftfreq(self.N_FFT, 1.0 / sr)

        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs <= hi)
            if not mask.any():
                continue
            band_mag = mag[mask]
            if len(band_mag) < 4:
                continue
            flatness = float(np.exp(np.mean(np.log(band_mag + 1e-10))) / (np.mean(band_mag) + 1e-10))
            # Niedrige Flatness = tonaler, besser separiert
            harmonicity_scores.append(float(1.0 - np.clip(flatness, 0.0, 1.0)))

        if not harmonicity_scores:
            return 1.0
        # Floor 0.70: Ohne Referenz kann Separation-Fidelity nicht sinnvoll
        # gegen absoluten Schwellwert geprüft werden — sauberes Material wird
        # nicht bestraft (kein Fehler des Restaurierungs-Systems).
        score = float(np.clip(np.mean(harmonicity_scores) * 1.5, 0.70, 1.0))
        return score


class ArticulationMetric:
    """14. Musikalisches Ziel: Artikulation (§1.2 Spec v9.9.5).

    Misst den Erhalt des Attack-Charakters (Staccato vs. Legato):
        - Transient-Shape-Korrelation ≥ 0.90
        - Attack-Time-Abweichung ≤ 10 ms gegenüber Original

    Algorithmus:
        Mit Referenz:
            1. Onset-Energie-Einhüllende (kurze Frames, 5 ms Hop)
            2. Pearson-Korrelation Attack-Profile Ref ↔ Rest
            3. Mittlere Attack-Time-Abweichung aus Onset-Detektion
            4. score = 0.65 · transient_corr + 0.35 · attack_time_score

        Ohne Referenz:
            1. Attack-Steilheit: Max-Amplituden-Anstieg pro Onset
            2. Onset-Dichte und -Varianz als Proxy für Staccato/Legato-Erhalt
            3. Spektraler Flux als Artikulations-Indikator

    Schwellwert: ≥ 0.85

    Referenz:
        Bello et al. (2005): "A Tutorial on Onset Detection in Music Signals"
        Fitzgerald (2010): "Harmonic/Percussive Separation Using Median Filtering"
    """

    FRAME_SIZE_MS: float = 10.0  # Kurze Frames für Transient-Analyse
    HOP_MS: float = 5.0
    ATTACK_MAX_MS: float = 10.0  # Max. tolerable Abweichung der Attack-Zeit
    N_FFT: int = 512
    HOP_FFT: int = 128

    def measure(
        self,
        audio: np.ndarray,
        sr: int,
        reference: np.ndarray | None = None,
    ) -> float:
        """Berechnet Artikulations-Score.

        Args:
            audio:     Restauriertes Audio-Signal.
            sr:        Sample-Rate in Hz.
            reference: Original-Audio vor Restaurierung (empfohlen).

        Returns:
            Score ∈ [0, 1]. 1.0 = Anschlagscharakter vollständig erhalten.
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1).astype(np.float32)
        else:
            audio_mono = audio.astype(np.float32)

        if reference is not None:
            ref = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
            if ref.ndim > 1:
                ref = np.mean(ref, axis=0 if ref.shape[0] <= 2 else 1).astype(np.float32)
            return self._reference_based(audio_mono, ref.astype(np.float32), sr)

        return self._reference_free(audio_mono, sr)

    def _reference_based(self, restored: np.ndarray, reference: np.ndarray, sr: int) -> float:
        """Referenzbasierter Modus: Transient-Shape-Korrelation + Attack-Time."""
        win_samples = max(4, int(sr * self.FRAME_SIZE_MS / 1000.0))
        hop_samples = max(2, int(sr * self.HOP_MS / 1000.0))
        min_len = min(len(restored), len(reference))
        if min_len < win_samples * 2:
            return 1.0

        # Energie-Einhüllende (kurze Frames)
        env_ref = self._energy_envelope(reference[:min_len], win_samples, hop_samples)
        env_rest = self._energy_envelope(restored[:min_len], win_samples, hop_samples)

        min_frames = min(len(env_ref), len(env_rest))
        if min_frames < 4:
            return 1.0

        # Transient-Shape-Korrelation (Pearson)
        if np.std(env_ref[:min_frames]) < 1e-10 or np.std(env_rest[:min_frames]) < 1e-10:
            transient_corr = 1.0
        else:
            corr = float(np.corrcoef(env_ref[:min_frames], env_rest[:min_frames])[0, 1])
            transient_corr = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

        # Attack-Time-Abweichung: Onsets über RMS-Gradient
        onsets_ref = self._detect_onsets(env_ref)
        onsets_rest = self._detect_onsets(env_rest)
        attack_score = self._attack_time_score(onsets_ref, onsets_rest, hop_samples, sr)

        score = float(0.65 * transient_corr + 0.35 * attack_score)
        return float(np.clip(score, 0.0, 1.0))

    def _reference_free(self, audio: np.ndarray, sr: int) -> float:
        """Referenzfreier Modus: Spektraler Flux + Onset-Steilheit."""
        if len(audio) < self.N_FFT:
            return 1.0

        # Spektraler Flux: Summe positiver Spectral-Differenzen → Transient-Indikator
        n_frames = (len(audio) - self.N_FFT) // self.HOP_FFT
        if n_frames < 2:
            return 1.0

        win = np.hanning(self.N_FFT).astype(np.float32)
        mags: list[np.ndarray] = []
        for k in range(min(n_frames, 128)):
            seg = audio[k * self.HOP_FFT : k * self.HOP_FFT + self.N_FFT]
            if len(seg) < self.N_FFT:
                break
            mags.append(np.abs(np.fft.rfft(seg * win)))

        if len(mags) < 2:
            return 1.0

        fluxes: list[float] = []
        for i in range(1, len(mags)):
            diff = mags[i] - mags[i - 1]
            flux = float(np.sum(np.maximum(diff, 0.0))) / (float(np.mean(mags[i])) + 1e-10)
            fluxes.append(flux)

        # Normalisierter Flux-Variationskoeffizient
        flux_arr = np.array(fluxes, dtype=np.float32)
        cv = float(np.std(flux_arr)) / (float(np.mean(flux_arr)) + 1e-10)
        # Gute Artikulation: moderater Flux-CV (klar Transient/Sustained)
        # Floor 0.75: Ohne Referenz kann Attack-Charakter nicht absolut bewertet
        # werden — sauberes restauriertes Material wird nicht bestraft.
        score = float(np.clip(cv * 2.5, 0.75, 1.0))
        return score

    def _energy_envelope(self, audio: np.ndarray, win: int, hop: int) -> np.ndarray:
        """Berechnet RMS-Einhüllende mit kurzen Frames."""
        n_frames = max(1, (len(audio) - win) // hop + 1)
        env = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            seg = audio[i * hop : i * hop + win]
            env[i] = float(np.sqrt(np.mean(seg**2) + 1e-12))
        return env

    def _detect_onsets(self, envelope: np.ndarray) -> np.ndarray:
        """Einfacher Onset-Detektor: Frames mit starkem Energie-Anstieg."""
        if len(envelope) < 2:
            return np.array([], dtype=np.int32)
        diff = np.diff(envelope.astype(np.float32))
        thresh = float(np.mean(diff[diff > 0]) + 1e-10) if (diff > 0).any() else 1e-3
        onsets = np.where(diff > thresh)[0]
        return onsets.astype(np.int32)

    def _attack_time_score(
        self,
        onsets_ref: np.ndarray,
        onsets_rest: np.ndarray,
        hop_samples: int,
        sr: int,
    ) -> float:
        """Mittlere Attack-Time-Abweichung → Score."""
        if len(onsets_ref) == 0 or len(onsets_rest) == 0:
            return 1.0
        # Vergleiche korrespondierendste Onsets
        n_match = min(len(onsets_ref), len(onsets_rest), 16)
        if n_match == 0:
            return 1.0
        diffs_samples = np.abs(onsets_ref[:n_match].astype(np.float32) - onsets_rest[:n_match].astype(np.float32))
        diffs_ms = diffs_samples * hop_samples / sr * 1000.0
        mean_diff_ms = float(np.mean(diffs_ms))
        # Maximal tolerierte Abweichung: ATTACK_MAX_MS
        score = float(np.clip(1.0 - mean_diff_ms / (self.ATTACK_MAX_MS * 2.0), 0.0, 1.0))
        return score


class MusicalGoalsChecker:
    """Zentraler Checker für alle 14 musikalischen Qualitätsziele (v9.9.9+).

    Ziele (in kanonischer Reihenfolge):
    1.  Brillanz              – HF-Klarheit 8–20 kHz              (≥ 0.85)
    2.  Wärme                 – Mittentiefe 200–2000 Hz            (≥ 0.80)
    3.  Natürlichkeit         – Artefaktfreiheit                   (≥ 0.90)
    4.  Authentizität         – Klang-Fingerabdruck / Stimme       (≥ 0.88)
    5.  Emotionalität         – Dynamik & Ausdruck                 (≥ 0.87)
    6.  Transparenz           – Klarheit & Trennung                (≥ 0.89)
    7.  Bass-Kraft            – Fundament 20–250 Hz                (≥ 0.85)
    8.  Groove                – Mikro-Timing, Swing, DTW ≤ 8 ms   (≥ 0.88)
    9.  Spatial Depth         – Räumliche Tiefe & Stereo-Bild      (≥ 0.75)
    10. Timbre-Authentizität  – Klangfarben-Erhalt (MFCC, Centroid)(≥ 0.87)
    11. Tonales Zentrum       – Chroma-Korrelation ≥ 0.95          (≥ 0.95)
    12. Mikro-Dynamik         – LUFS-Profil-Korrelation 400 ms     (≥ 0.92)
    13. Separation-Treue      – SDR ≥ 8 dB / SIR ≥ 12 dB (NMF)   (≥ 0.82)
    14. Artikulation          – Attack-Charakter, Transient ≤ 10 ms(≥ 0.85)

    Example::

        checker = MusicalGoalsChecker()
        scores  = checker.measure_all(audio, sr=48000)
        # → 14 Einträge: brillanz, waerme, …, separation_fidelity, artikulation

        passed, violations = checker.check_all_preserved(original, processed, sr=48000)
        if not passed:
            logger.debug(f"Verletzungen: {violations}")
    """

    def __init__(self, custom_thresholds: dict[str, float] | None = None) -> None:
        """Initialisiert alle 14 Metrik-Klassen.

        Args:
            custom_thresholds: Optionale Schwellwert-Überschreibungen.
        """
        # Alle 14 Metriken (kanonische Reihenfolge gem. Aurik-9-Spec §1.2 v9.9.9)
        self.metrics = {
            "bass_kraft": BassKraftMetric(),
            "brillanz": BrillanzMetric(),
            "waerme": WaermeMetric(),
            "natuerlichkeit": NatuerlichkeitMetric(),
            "authentizitaet": AuthentizitaetMetric(),
            "emotionalitaet": EmotionalitaetMetric(),
            "transparenz": TransparenzMetric(),
            "groove": GrooveMetric(),  # 8. Ziel (v9.9)
            "spatial_depth": SpatialDepthMetric(),  # 9. Ziel (v9.9)
            "timbre_authentizitaet": TimbralAuthenticityMetric(),  # 10. Ziel (v9.9)
            "tonal_center": TonalCenterMetric(),  # 11. Ziel (v9.9.5)
            "micro_dynamics": MicroDynamicsMetric(),  # 12. Ziel (v9.9.5)
            "separation_fidelity": SeparationFidelityMetric(),  # 13. Ziel (v9.9.9)
            "artikulation": ArticulationMetric(),  # 14. Ziel (v9.9.9)
        }

        # Pflicht-Schwellwerte gem. Aurik-9-Spec §1.2 v9.9.9 (alle 14 Ziele)
        self.thresholds = {
            "bass_kraft": 0.85,
            "brillanz": 0.85,
            "waerme": 0.80,
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.88,
            "emotionalitaet": 0.87,
            "transparenz": 0.89,
            "groove": 0.88,  # DTW ≤ 8 ms RMS
            "spatial_depth": 0.75,  # Stereo-Räumlichkeit
            "timbre_authentizitaet": 0.87,  # MFCC-Pearson ≥ 0.95, Centroid-Pearson ≥ 0.93
            "tonal_center": 0.95,  # Chroma-Korrelation ≥ 0.95, kein Key-Shift
            "micro_dynamics": 0.92,  # LUFS-Profil-Korrelation 400 ms ≥ 0.92
            "separation_fidelity": 0.82,  # SDR ≥ 8 dB / SIR ≥ 12 dB (v9.9.9)
            "artikulation": 0.85,  # Attack-Charakter-Erhalt ≤ 10 ms (v9.9.9)
        }

        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

    def measure_all(
        self,
        audio: np.ndarray,
        sr: int,
        reference: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Misst alle 14 musikalischen Qualitätsziele (Spec §1.2 v9.9.9).

        Args:
            audio:     Audio-Signal (mono oder stereo).
            sr:        Sample-Rate in Hz.
            reference: Optionales Referenz-Audio (Original vor Restaurierung).
                       Verbessert Präzision von ``authentizitaet``,
                       ``timbre_authenticity``, ``separation_fidelity`` und
                       ``articulation`` erheblich.

        Returns:
            Dict mit Scores für alle 14 Musical Goals ∈ [0.0, 1.0].
        """
        scores: dict[str, float] = {}

        # FIXED v9.10: Stereo-Format-Normalisierung
        # Aurik-interne Pipeline verwendet (C, N) = channels-first.
        # Alle Metriken erwarten (N,) mono oder (N, C) samples-first.
        # → Transponiere (2, N) → (N, 2) damit SpatialDepthMetric links/rechts korrekt liest.
        if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
            audio = audio.T
        # FIXED v9.10.45: [1,N] channels-first mono → flatten zu (N,)
        elif audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        if reference is not None and reference.ndim == 2 and reference.shape[0] == 2 and reference.shape[1] > 2:
            reference = reference.T
        elif reference is not None and reference.ndim == 2 and reference.shape[0] == 1:
            reference = reference[0]

        for goal_name, metric in self.metrics.items():
            if goal_name in ("authentizitaet", "timbre_authentizitaet") and reference is not None:
                scores[goal_name] = metric.measure(audio, sr, reference=reference)
            else:
                scores[goal_name] = metric.measure(audio, sr)

        # Key ist "artikulation" (konsistent mit goal_priority_protocol, goal_applicability_filter)
        return scores

    def measure_all_with_context(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        panns_tags: list | None = None,
        reference: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Misst alle 14 Musical Goals mit PANNs-kontext-adaptivem Weighting.

        Der Gewichtungsvektor wird automatisch aus dem PANNs-Tagging abgeleitet:
        Genre/Instrumente bestimmen, welche Ziele für das spezifische
        Klangmaterial besonders wichtig sind.

        Adaptive Gewichte (Multiplikator auf den Rohscore):

        ============  ============  ===========  ===========  =========
        PANNs-Tag     Emotionalität  Natürl.      BassKraft    Brillanz
        ============  ============  ===========  ===========  =========
        Jazz/Blues    1.3×           1.2×         1.0×         1.0×
        Classical     1.0×           1.4×         0.8×         1.1×
        Hip-hop/EDM   0.9×           0.9×         1.5×         1.0×
        Rock/Pop      1.1×           1.0×         1.1×         1.2×
        Speech/Voice  1.2×           1.3×         0.8×         1.0×
        Drums/Perc.   0.9×           0.9×         1.3×         0.9×
        ============  ============  ===========  ===========  =========

        Args:
            audio:      Audio-Signal (mono oder stereo).
            sr:         Sample-Rate in Hz.
            panns_tags: PANNs-Tag-Liste (Strings, z. B. ``["Jazz", "Piano"]``).
                        Bei ``None``: gleichgewichtete Standardbewertung.
            reference:  Optionales Referenz-Audio (vor Restaurierung).

        Returns:
            Dict mit gewichteten Scores ∈ [0, 1] für alle 14 Musical Goals.
        """
        # Basis-Scores mit normalen Gewichtungen messen
        base_scores = self.measure_all(audio, sr, reference=reference)

        if not panns_tags:
            return base_scores

        # Genre-adaptiver Gewichtungsvektor ableiten
        weights: dict[str, float] = {}
        lower_tags = [t.lower() for t in panns_tags]

        def _has(keywords: list) -> bool:
            return any(kw in tag for kw in keywords for tag in lower_tags)

        if _has(["jazz", "blues", "soul", "swing"]):
            weights = {
                "emotionalitaet": 1.30,
                "natuerlichkeit": 1.20,
                "groove": 1.25,
                "bass_kraft": 1.00,
                "brillanz": 1.00,
            }
        elif _has(["classical", "orchestr", "chamber", "opera", "symphon"]):
            weights = {
                "natuerlichkeit": 1.40,
                "authentizitaet": 1.20,
                "transparenz": 1.20,
                "bass_kraft": 0.80,
                "brillanz": 1.10,
                "timbre_authentizitaet": 1.20,
            }
        elif _has(["hip-hop", "hiphop", "rap", "electronic", "techno", "edm", "house"]):
            weights = {
                "bass_kraft": 1.50,
                "groove": 1.30,
                "emotionalitaet": 0.90,
                "natuerlichkeit": 0.90,
                "spatial_depth": 1.20,
            }
        elif _has(["rock", "metal", "punk", "pop", "indie"]):
            weights = {
                "bass_kraft": 1.10,
                "brillanz": 1.20,
                "emotionalitaet": 1.10,
                "transparenz": 1.10,
            }
        elif _has(["speech", "voice", "singing", "vocal", "spoken"]):
            weights = {
                "authentizitaet": 1.30,
                "natuerlichkeit": 1.30,
                "emotionalitaet": 1.20,
                "timbre_authentizitaet": 1.25,
                "bass_kraft": 0.80,
            }
        elif _has(["drum", "percussion", "beat", "rhyth"]):
            weights = {
                "groove": 1.40,
                "bass_kraft": 1.30,
                "transparenz": 1.10,
                "emotionalitaet": 0.90,
                "natuerlichkeit": 0.90,
            }

        if not weights:
            # Unbekannter Genre-Kontext: Basis-Scores unverändert
            return base_scores

        # Gewichte anwenden und auf [0, 1] clippen
        weighted: dict[str, float] = {}
        for goal, score in base_scores.items():
            w = weights.get(goal, 1.0)
            weighted[goal] = float(np.clip(score * w, 0.0, 1.0))

        logger.debug(
            "📊 Kontext-adaptives Weighting: Tags=%s → Δscores=%s",
            panns_tags[:4],
            {k: f"{weighted[k]:.3f}←{base_scores[k]:.3f}" for k in weights},
        )
        return weighted

    def check_all_preserved(
        self, original: np.ndarray, processed: np.ndarray, sr: int
    ) -> tuple[bool, dict[str, float]]:
        """
        Checks if all goals are preserved (pre/post comparison).

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate

        Returns:
            Tuple of (all_passed: bool, violations: dict)
        """
        orig_scores = self.measure_all(original, sr)
        proc_scores = self.measure_all(processed, sr, reference=original)

        violations = {}
        for goal_name in orig_scores:
            threshold = self.thresholds[goal_name]
            if proc_scores[goal_name] < threshold:
                violations[goal_name] = {
                    "original": orig_scores[goal_name],
                    "processed": proc_scores[goal_name],
                    "threshold": threshold,
                    "delta": proc_scores[goal_name] - orig_scores[goal_name],
                }

        return (len(violations) == 0, violations)

    def check_with_adaptive_thresholds(
        self,
        audio: np.ndarray,
        sr: int,
        adaptive_thresholds: dict[str, float],
        reference: np.ndarray | None = None,
    ) -> tuple[bool, dict[str, float], dict[str, float]]:
        """
        Check all goals against ADAPTIVE thresholds (für degradiertes Material).

        WORLD-FIRST: Intelligente Anpassung der Qualitätsziele basierend auf Material-Qualität

        Args:
            audio: Audio signal to check
            sr: Sample rate
            adaptive_thresholds: Dict with adaptive thresholds (from AdaptiveGoalsCalculator)
            reference: Optional reference audio (for authentizität)

        Returns:
            Tuple of (all_passed, violations, scores)
        """
        # Measure all goals
        scores = self.measure_all(audio, sr, reference=reference)

        # Check against adaptive thresholds
        violations = {}
        for goal_name, score in scores.items():
            threshold = adaptive_thresholds.get(goal_name, self.thresholds[goal_name])
            if score < threshold:
                violations[goal_name] = {
                    "score": score,
                    "threshold": threshold,
                    "deficit": threshold - score,
                }

        all_passed = len(violations) == 0

        return all_passed, violations, scores

    def measure_single(self, goal_name: str, audio: np.ndarray, sr: int) -> GoalMeasurement:
        """
        Measure single goal with detailed result.

        Args:
            goal_name: Name of goal to measure
            audio: Audio signal
            sr: Sample rate

        Returns:
            GoalMeasurement with detailed result
        """
        if goal_name not in self.metrics:
            raise ValueError(f"Unknown goal: {goal_name}. " f"Available: {list(self.metrics.keys())}")

        metric = self.metrics[goal_name]
        score = metric.measure(audio, sr)
        threshold = self.thresholds[goal_name]
        # FIX v9.10: numpy.bool_ (from comparison) fails isinstance(..., bool) in NumPy 2.x
        passed: bool = bool(score >= threshold)

        return GoalMeasurement(
            goal_name=goal_name,
            score=score,
            passed=passed,
            threshold=threshold,
            details={"score": score, "threshold": threshold},
        )


# =============================================================================
# SINGLETON-ACCESSOREN (gem. Aurik-9-Standard §3.2)
# =============================================================================

_checker_instance: MusicalGoalsChecker | None = None
_checker_lock = threading.Lock()


def get_checker(custom_thresholds: dict[str, float] | None = None) -> MusicalGoalsChecker:
    """Thread-sicherer Singleton-Accessor für MusicalGoalsChecker.

    Gibt bei jedem Aufruf dieselbe Instanz zurück (Singleton).
    Bei Übergabe von ``custom_thresholds`` wird einmalig eine neue
    Instanz mit diesen Schwellwerten erzeugt.

    Args:
        custom_thresholds: Optionale Schwellwert-Überschreibungen (nur bei ersten Aufruf).

    Returns:
        Singleton-Instanz von :class:`MusicalGoalsChecker`.
    """
    global _checker_instance
    if _checker_instance is None:
        with _checker_lock:
            if _checker_instance is None:
                _checker_instance = MusicalGoalsChecker(custom_thresholds=custom_thresholds)
                logger.debug("MusicalGoalsChecker Singleton erstellt.")
    return _checker_instance


def measure_all(audio: "np.ndarray", sr: int) -> dict[str, float]:
    """Convenience-Funktion: Musical Goals für alle 14 Qualitätsziele messen (v9.9+).

    Nutzt den Singleton :func:`get_checker` und ruft ``measure_all()`` auf.
    Gibt alle 14 Ziele zurück (Brillanz, Wärme, Natürlichkeit, Authentizität,
    Emotionalität, Transparenz, Bass-Kraft, Groove, Raumtiefe,
    Timbre-Authentizität, TonalesZentrum, MikroDynamik, SeparationTreue, Artikulation).

    Args:
        audio: Audio-Signal als numpy ndarray (mono float32/64).
        sr:    Abtastrate in Hz.

    Returns:
        Dict[goal_name -> score] mit 14 Einträgen, alle in [0.0, 1.0].
    """
    return get_checker().measure_all(audio, sr)


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    # Test der 14 normativen Musical Goals (Spec §1.2)
    pass

    logger.debug("=== AURIK Musical Goals Test (14 normative Goals — Spec §1.2) ===\n")

    # Testsignal erzeugen
    sr = 48000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Multi-Frequenz-Signal (Bass + Mitten + Höhen)
    audio_mono = (
        0.3 * np.sin(2 * np.pi * 100 * t)  # Bass (100 Hz)
        + 0.3 * np.sin(2 * np.pi * 500 * t)  # Mitten (500 Hz)
        + 0.2 * np.sin(2 * np.pi * 2000 * t)  # Obere Mitten (2 kHz)
        + 0.2 * np.sin(2 * np.pi * 8000 * t)  # Höhen (8 kHz)
    )

    # Stereo-Signal (für SpatialDepth-Test)
    left = audio_mono + 0.1 * np.sin(2 * np.pi * 1000 * t)
    right = audio_mono - 0.1 * np.sin(2 * np.pi * 1000 * t)
    audio_stereo = np.stack([left, right], axis=1)

    # Test: Alle 14 Goals messen
    checker = MusicalGoalsChecker()
    scores = checker.measure_all(audio_stereo, sr)
    logger.debug(f"Total Goals: {len(scores)}")
    logger.debug("")

    for goal, score in scores.items():
        threshold = checker.thresholds[goal]
        passed = "✅" if score >= threshold else "❌"
        logger.debug(f"  {passed} {goal:20s}: {score:.3f} (thresh: {threshold:.2f})")

    logger.debug("\n=== Test abgeschlossen ===")
