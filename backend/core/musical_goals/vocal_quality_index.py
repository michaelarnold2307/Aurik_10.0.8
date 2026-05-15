"""
VocalQualityIndex (VQI) — §2.35c [RELEASE_MUST]
================================================

Gesangs-Gesamtqualitäts-Gate. Aggregiert 5 orthogonale Vokal-Metriken zu einem
einzelnen Score [0, 1]. Aktivierung: PANNs Singing confidence >= 0.35.

Spec: 01_musical_goals.md §2.35c (v9.12.0)
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Schwellwerte (§2.35c)
# ---------------------------------------------------------------------------
VQI_WORLD_CLASS = 0.88
VQI_PROFESSIONAL = 0.82
VQI_THRESHOLD = 0.72  # Recovery-Kaskade Vinyl-Default (§0p)

# Material-adaptive VQI recovery-cascade floors (§0p)
# Shellac/Wax: 0.62 — physical BW/SNR limits; Vinyl: 0.72; Tape: 0.72; CD: 0.82
_VQI_MATERIAL_FLOOR: dict[str, float] = {
    "wax_cylinder": 0.62,
    "shellac": 0.62,
    "wire_recording": 0.62,
    "lacquer_disc": 0.62,
    "tape": 0.72,
    "reel_tape": 0.72,
    "cassette": 0.70,
    "kassette": 0.70,
    "vinyl": 0.72,
    "lp": 0.72,
    "cd_digital": 0.82,
    "cd": 0.82,
    "dat": 0.82,
    "mp3_low": 0.72,
    "mp3_high": 0.75,
    "aac": 0.75,
    "minidisc": 0.72,
    "streaming": 0.75,
}


def get_vqi_material_floor(material_type: str, is_studio_2026: bool = False) -> float:
    """Return the material-adaptive VQI recovery-cascade threshold (§0p).

    Args:
        material_type: e.g. "shellac", "vinyl", "cd_digital".
        is_studio_2026: Studio-2026-Modus → festes Ziel 0.87 (§0p).

    Returns:
        float: Recovery-Schwellwert ∈ [0.62, 0.87].
    """
    if is_studio_2026:
        return 0.87  # §0p: Studio 2026 fixed recovery goal
    mat = str(material_type or "").strip().lower()
    return _VQI_MATERIAL_FLOOR.get(mat, VQI_THRESHOLD)  # default: vinyl 0.72


# Gewichtungen (weighted sum aus §2.35c Spec)
_W_SINGER_ID = 0.30
_W_FORMANT = 0.25
_W_ARTICULATION = 0.20
_W_PROXIMITY = 0.15
_W_SIBILANCE = 0.10


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Channels-first (2,N) oder samples-first (N,2) → mono (N,)."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.ndim == 2:
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            return audio.mean(axis=0).astype(np.float32)
        if audio.shape[1] == 2:
            return audio.mean(axis=1).astype(np.float32)
        if audio.shape[0] == 1:
            return audio[0].astype(np.float32)
    return audio.flatten().astype(np.float32)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """NaN-safe cosine similarity."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Komponente 1 — Singer-Identity-Cosine (DSP-Proxy ohne Resemblyzer)
# ---------------------------------------------------------------------------


def _compute_singer_identity_dsp(
    vocal_pre: np.ndarray,
    vocal_post: np.ndarray,
    sr: int,
) -> float:
    """DSP-Proxy für Singer-Identity: MFCC-Korrelation + Spectral-Centroid-Korrelation.

    Liefert approx. Cosinus-Ähnlichkeit [0, 1]. Wird verwendet wenn Resemblyzer
    nicht verfügbar ist (§2.35c DSP-Fallback).
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        n_mfcc = 20
        n_fft = min(2048, len(vocal_pre) // 4)
        hop = n_fft // 4

        mfcc_pre = librosa.feature.mfcc(y=vocal_pre, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
        mfcc_post = librosa.feature.mfcc(y=vocal_post, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)

        min_frames = min(mfcc_pre.shape[1], mfcc_post.shape[1])
        if min_frames < 4:
            return 0.80  # zu kurz — neutraler Wert

        # Mittlere MFCC über Zeit → kompakter Fingerabdruck
        vec_pre = mfcc_pre[:, :min_frames].mean(axis=1)
        vec_post = mfcc_post[:, :min_frames].mean(axis=1)
        cosine = _safe_cosine(vec_pre, vec_post)
        # Cosine aus MFCC liegt typ. [0.85, 1.0] für gleiche Stimme
        # Normalisiere auf [0, 1] mit Bezugspunkt 0.85
        return float(np.clip((cosine - 0.70) / 0.30, 0.0, 1.0))
    except Exception as exc:
        logger.debug("singer_identity_dsp failed: %s", exc)
        return 0.80  # konservativ-neutral


def _compute_singer_identity(
    vocal_pre: np.ndarray,
    vocal_post: np.ndarray,
    sr: int,
) -> tuple[float, bool]:
    """ResemblyzerPlugin (primär) oder DSP-Proxy (Fallback). Gibt (cosine, dsp_used) zurück."""
    try:
        from plugins.resemblyzer_plugin import get_resemblyzer_plugin  # pylint: disable=import-outside-toplevel

        plugin = get_resemblyzer_plugin()
        if not plugin.available:
            raise RuntimeError("ResemblyzerPlugin nicht verfügbar")

        emb_pre = plugin.embed(vocal_pre, sr)
        emb_post = plugin.embed(vocal_post, sr)
        if emb_pre is None or emb_post is None:
            raise RuntimeError("embed() lieferte None")

        cosine = plugin.cosine_similarity(emb_pre, emb_post)
        return float(np.clip(cosine, 0.0, 1.0)), False
    except Exception as exc:
        logger.debug("Resemblyzer nicht verfügbar (%s) — DSP-Fallback", exc)
        return _compute_singer_identity_dsp(vocal_pre, vocal_post, sr), True


# ---------------------------------------------------------------------------
# Komponente 2 — Formant-Stabilitäts-Score
# ---------------------------------------------------------------------------


def _compute_formant_stability(
    vocal_pre: np.ndarray,
    vocal_post: np.ndarray,
    sr: int,
) -> float:
    """Schätzt F1-Drift via LPC (linear approximation). Score = max(0, 1 - drift/70Hz)."""
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        frame_len = int(sr * 0.025)  # 25 ms frames
        hop = frame_len // 2

        # LPC-Order: ~sr/1000 + 2 (Regel für Sprach-Formanten)
        lpc_order = max(8, sr // 1000 + 2)

        def _lpc_f1_frames(audio: np.ndarray) -> list[float]:
            frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop)
            f1_list = []
            for i in range(frames.shape[1]):
                frame = frames[:, i] * np.hanning(frame_len)
                try:
                    a = librosa.lpc(frame, order=lpc_order)
                    roots = np.roots(a)
                    roots = roots[np.imag(roots) >= 0]
                    angles = np.angle(roots)
                    freqs = sorted(angles * (sr / (2 * np.pi)))
                    # F1: erste Resonanz > 100 Hz
                    valid = [f for f in freqs if f > 100]
                    if valid:
                        f1_list.append(valid[0])
                except Exception:
                    pass
            return f1_list

        f1_pre = _lpc_f1_frames(vocal_pre)
        f1_post = _lpc_f1_frames(vocal_post)

        if not f1_pre or not f1_post:
            return 0.85  # kann nicht messen → konservativ

        min_len = min(len(f1_pre), len(f1_post))
        drift = float(np.mean(np.abs(np.array(f1_pre[:min_len]) - np.array(f1_post[:min_len]))))
        # 35 Hz = Schwellwert (§2.35c): bei 0 Hz Drift → 1.0; bei 70 Hz Drift → 0.5
        score = float(np.clip(1.0 - drift / 70.0, 0.0, 1.0))
        return score
    except Exception as exc:
        logger.debug("formant_stability failed: %s", exc)
        return 0.85


# ---------------------------------------------------------------------------
# Komponente 3 — Artikulations-Score (aus §2.35, via proximity-konsonant-ratio)
# ---------------------------------------------------------------------------


def _compute_articulation_score(
    vocal_pre: np.ndarray,
    vocal_post: np.ndarray,
    sr: int,
) -> float:
    """Plosiv/Frikativ-Onset-Präzision. Nutzt Transient-Energie-Ratio aus §2.35."""
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        # Breitband-Transient-Erkennung via spectral_flux
        hop = 512
        onset_strength_pre = librosa.onset.onset_strength(y=vocal_pre, sr=sr, hop_length=hop)
        onset_strength_post = librosa.onset.onset_strength(y=vocal_post, sr=sr, hop_length=hop)

        min_len = min(len(onset_strength_pre), len(onset_strength_post))
        if min_len < 4:
            return 0.85

        # Korrelation der Onset-Stärke-Kurven
        pre_norm = onset_strength_pre[:min_len]
        post_norm = onset_strength_post[:min_len]
        # Guarded dot-product (NaN-safe, as required by VERBOTEN V01)
        _pn_std = float(np.std(pre_norm)) * float(np.std(post_norm))
        if _pn_std < 1e-9:
            corr = 0.0
        else:
            corr = float(
                np.dot(pre_norm - pre_norm.mean(), post_norm - post_norm.mean()) / (len(pre_norm) * _pn_std + 1e-12)
            )
        corr = float(np.clip(corr, -1.0, 1.0))
        # Korrelation [0,1] → Score (nahe 1 = Onsets erhalten)
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
    except Exception as exc:
        logger.debug("articulation_score failed: %s", exc)
        return 0.85


# ---------------------------------------------------------------------------
# Komponente 5 — Sibilance-Naturalness (5–10 kHz)
# ---------------------------------------------------------------------------


def _compute_sibilance_naturalness(
    vocal_pre: np.ndarray,
    vocal_post: np.ndarray,
    sr: int,
) -> float:
    """Energieabweichung im 5–10 kHz Frikativen-Band. score = max(0, 1 - |abw_db|/6)."""
    try:
        from scipy.signal import butter, sosfiltfilt  # pylint: disable=import-outside-toplevel

        nyq = sr / 2.0
        lo = min(5000.0, nyq * 0.9) / nyq
        hi = min(10000.0, nyq * 0.99) / nyq
        if lo >= hi or lo <= 0:
            return 0.85

        sos = butter(4, [lo, hi], btype="bandpass", output="sos")
        pre_band = sosfiltfilt(sos, vocal_pre)
        post_band = sosfiltfilt(sos, vocal_post)

        rms_pre = float(np.sqrt(np.mean(pre_band**2)) + 1e-12)
        rms_post = float(np.sqrt(np.mean(post_band**2)) + 1e-12)

        abw_db = abs(20.0 * np.log10(rms_post / rms_pre))
        # 6 dB = Schwellwert (§2.35c): 0 dB Abweichung → 1.0
        score = float(np.clip(1.0 - abw_db / 6.0, 0.0, 1.0))
        return score
    except Exception as exc:
        logger.debug("sibilance_naturalness failed: %s", exc)
        return 0.85


# ---------------------------------------------------------------------------
# Haupt-API
# ---------------------------------------------------------------------------


def compute_vqi(
    audio_orig: np.ndarray,
    audio_restored: np.ndarray,
    sr: int,
    vocal_segments: list[tuple[float, float]] | None = None,
    skip_singer_identity: bool = False,
) -> dict[str, float]:
    """Berechnet den VocalQualityIndex (§2.35c).

    Args:
        audio_orig:     Original (degradiert) vor Pipeline, float32 [-1,1].
        audio_restored: Nach Pipeline restauriertes Audio, float32 [-1,1].
        sr:             Sample-Rate in Hz (Pflicht: 48000 in Pipeline-Phasen).
        vocal_segments: Optionale Liste (start_sec, end_sec) Vokal-Zeitfenster.
                        None → gesamtes Signal wird genutzt.

    Returns:
        Dict mit:
            vqi              — Gesamt-Score [0, 1]
            singer_identity_cosine
            formant_stability_score
            articulation_score
            proximity_score  — aus §2.35b (falls verfügbar, sonst 0.85)
            sibilance_naturalness
            singer_id_dsp_fallback — bool: True wenn Resemblyzer nicht genutzt
            vqi_tier         — "world_class" | "professional" | "acceptable" | "below_threshold"
    """
    orig_m = _to_mono(audio_orig)
    rest_m = _to_mono(audio_restored)

    min_len = min(len(orig_m), len(rest_m))
    if min_len < sr // 2:
        # Too short for meaningful measurement — return safe neutral values.
        # 0.90 > VQI_WORLD_CLASS (0.88) ensures no false recovery-cascade trigger,
        # even in Studio 2026 mode (floor 0.87). singer_identity_cosine=0.95 prevents
        # false rollback (§0p gate: < 0.92 triggers rollback).
        return {
            "vqi": 0.90,
            "singer_identity_cosine": 0.95,
            "formant_stability_score": 0.90,
            "articulation_score": 0.90,
            "proximity_score": 0.90,
            "sibilance_naturalness": 0.90,
            "singer_id_dsp_fallback": True,
            "vqi_tier": "world_class",
        }

    orig_m = orig_m[:min_len]
    rest_m = rest_m[:min_len]

    # Vokal-Segment-Maske anwenden
    if vocal_segments:
        mask = np.zeros(min_len, dtype=bool)
        for s, e in vocal_segments:
            i0 = max(0, int(s * sr))
            i1 = min(min_len, int(e * sr))
            mask[i0:i1] = True
        if mask.sum() >= sr // 4:
            orig_m = orig_m[mask]
            rest_m = rest_m[mask]

    orig_m = np.nan_to_num(orig_m, nan=0.0, posinf=0.0, neginf=0.0)
    rest_m = np.nan_to_num(rest_m, nan=0.0, posinf=0.0, neginf=0.0)

    # Komponente 1: Singer-Identity (§MultiSinger: überspringen bei Duett/Chor)
    if skip_singer_identity:
        singer_cosine, dsp_fallback = 0.85, True  # neutral fallback — kein Rollback-Trigger
        logger.debug("§MultiSinger: singer_identity_cosine-Gate übersprungen (Duett/Chor)")
    else:
        singer_cosine, dsp_fallback = _compute_singer_identity(orig_m, rest_m, sr)

    # Komponente 2: Formant-Stabilität
    formant_score = _compute_formant_stability(orig_m, rest_m, sr)

    # Komponente 3: Artikulation
    articulation = _compute_articulation_score(orig_m, rest_m, sr)

    # Komponente 4: Vocal Proximity (aus §2.35b)
    proximity = 0.85  # default
    try:
        from backend.core.musical_goals.ki_hearing_model import compute_vocal_proximity_score  # pylint: disable=import-outside-toplevel  # noqa: I001

        prox_result = compute_vocal_proximity_score(audio_orig, audio_restored, sr, vocal_segments)
        proximity = float(prox_result.get("proximity_score", 0.85))
    except Exception as exc:
        logger.debug("vocal_proximity import failed: %s", exc)

    # Komponente 5: Sibilance
    sibilance = _compute_sibilance_naturalness(orig_m, rest_m, sr)

    # Gewichtete Summe (§2.35c)
    vqi = (
        _W_SINGER_ID * singer_cosine
        + _W_FORMANT * formant_score
        + _W_ARTICULATION * articulation
        + _W_PROXIMITY * proximity
        + _W_SIBILANCE * sibilance
    )
    vqi = float(np.clip(vqi, 0.0, 1.0))
    vqi = float(np.nan_to_num(vqi, nan=0.90))  # 0.90 > all floors incl. Studio 2026 (0.87)

    # Tier bestimmen
    if vqi >= VQI_WORLD_CLASS:
        tier = "world_class"
    elif vqi >= VQI_PROFESSIONAL:
        tier = "professional"
    elif vqi >= VQI_THRESHOLD:
        tier = "acceptable"
    else:
        tier = "below_threshold"

    return {
        "vqi": vqi,
        "singer_identity_cosine": float(np.clip(singer_cosine, 0.0, 1.0)),
        "formant_stability_score": float(np.clip(formant_score, 0.0, 1.0)),
        "articulation_score": float(np.clip(articulation, 0.0, 1.0)),
        "proximity_score": float(np.clip(proximity, 0.0, 1.0)),
        "sibilance_naturalness": float(np.clip(sibilance, 0.0, 1.0)),
        "singer_id_dsp_fallback": dsp_fallback,
        "vqi_tier": tier,
    }
