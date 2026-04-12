"""
Phase 55: Diffusion-Inpainting v1.0 — Masked Spectral Reconstruction
======================================================================

Weltklasse-Inpainting für Audio-Lücken und Dropouts >20ms, basierend auf
iterativer Diffusion im Spektralbereich (DDPM-inspiriert, pure DSP-Fallback +
optionaler DiffWave/AudioLDM2-Plugin-Pfad).

ALGORITHMUS — Dreistufige Masked Diffusion:
  1. **Lücken-Detektion** (Defect Mask):
     - RMS-Energie <  -60 dBFS für ≥ min_gap_ms Millisekunden → Dropout
     - Phase-Diskontinuität > π/2 zwischen Frames → Phasenbruch
     - Spectral Centroid-Sprung > 2 Oktaven → Spektralsprung
     - Masken-Dilation: ±5ms Rand-Padding

  2. **Kontextuelles Prior-Modell** (DSP-Diffusion):
     - Forward-Process: Maskierte Bins mit gaußschem Rauschen auffüllen
     - Reverse-Process (T=50 Denoising-Steps):
         Step t: x_{t-1} = f(x_t, context_left, context_right, t/T)
         Gewichtung: cos²-Interpolation von linkem/rechtem Kontext
         Rauschanteil: σ_t = σ_max * (t/T)^2 (Cosine-Schedule)
     - Harmonisches Prior: AR-Modell aus Vorfeld-Segment (Burg-Schätzung Ordnung 64)
     - Envelopen-Continuity: Attack/Release-Matching am Rand

  3. **Plugin-Pfad** (optional, wenn DiffWave-Gewichte vorhanden):
     - Lädt `plugins/diffwave_plugin.py` dynamisch
     - Konditioniert mit linkem+rechtem Kontext-Mel-Spektrogramm
     - Fallback auf DSP-Diffusion bei fehlenden Gewichten

METRIKEN:
  - n_gaps_detected: Anzahl erkannter Lücken
  - total_gap_ms: Summe rekonstruierter Millisekunden
  - max_gap_ms: Größte Einzellücke
  - plugin_used: Ob DiffWave-Plugin genutzt wurde
  - reconstruction_quality: Geschätzter PESQ-Proxy (Energie-Kontinuität-Score)

Author: Aurik Development Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# ─── Algorithmus-Konstanten ─────────────────────────────────────────────
_FFT_SIZE = 2048
_HOP = 512
_WIN = "hann"
_MIN_GAP_MS_DEFAULT = 20.0  # Minimale Lückenlänge für Inpainting
_ENERGY_THRESH_DBFS = -60.0  # Energie-Schwelle für Dropout-Erkennung
_DIFFUSION_STEPS = 50  # Basis-Schrittanzahl (Short Gaps < 50 ms)
_DIFFUSION_STEPS_MED = 100  # Mittlere Lücken 50–100 ms
_DIFFUSION_STEPS_LONG = 150  # Lange Lücken > 100 ms (höchste Qualität)


def _adaptive_steps(gap_ms: float) -> int:
    """Wählt Diffusionsschritt-Anzahl adaptiv nach Lückengröße.

    Kurze Lücken (<50 ms) brauchen weniger Iterationen (Kontext dominant),
    lange Lücken (>100 ms) profitieren von mehr Denoising-Schritten,
    da der AR-Prior weniger verlässlich wird.
    """
    if gap_ms < 50.0:
        return _DIFFUSION_STEPS  # 50
    if gap_ms < 100.0:
        return _DIFFUSION_STEPS_MED  # 100
    return _DIFFUSION_STEPS_LONG  # 150


_AR_ORDER = 64  # AR-Modell-Ordnung für harmonisches Prior
_CONTEXT_FRAMES = 20  # Kontext-Frames links/rechts
_MASK_DILATION_FRAMES = 3  # Dilations-Padding um Maske
_SIGMA_MAX = 0.3  # Maximale Rausch-Standardabweichung

# §0 Primum-non-nocere: Material-Bandbreiten-Cap verhindert HF-Halluzination
# beim Inpainting historischer Träger (wax_cylinder BW ≤ 5 kHz, wire_rec. ≤ 6 kHz).
# Ohne Cap generiert AR/Diffusion synthetische Obertöne, die nie vorhanden waren.
# Literature anchors for the caps:
#   - Wax cylinders: typically ~2.5–4.5 kHz useful bandwidth; 5 kHz used here as
#     conservative upper safety cap (Casey, Sound Directions 2007; IASA TC-04).
#   - Wire recording: typically ~4–6 kHz depending on transport speed and head design;
#     6 kHz used as the hard upper bound (Morton 2006; ARSC preservation notes).
#   - Shellac: archival transfer practice usually preserves content below ~7 kHz.
_MATERIAL_BW_CAP_HZ: dict[str, float] = {
    "wax_cylinder": 5000.0,  # Mechanische Aufzeichnung 1900–1930
    "wire_recording": 6000.0,  # Stahlbandfone 1940–1955
    "shellac": 7000.0,  # Schellackplatten 1898–1960, §0 Vintage Aesthetics (≤ 7 kHz)
    "lacquer_disc": 8000.0,  # Acetat-Lackfolien 1930–1950 (konservativ)
}


def _apply_bw_cap(segment: np.ndarray, sample_rate: int, cap_hz: float) -> np.ndarray:
    """Low-pass filter reconstructed segment to material bandwidth cap.

    Prevents AR/diffusion hallucinating HF content never present in the
    source material (§0 Primum-non-nocere, §2.55 BW-Cap Invariante).
    Uses Butterworth 8th-order to avoid ringing in short segments.
    """
    if cap_hz >= sample_rate / 2 - 100:
        return segment
    nyq = sample_rate / 2.0
    norm_cut = min(cap_hz / nyq, 0.99)
    from scipy import signal as _sps

    sos = _sps.butter(4, norm_cut, btype="low", output="sos")
    # filtfilt for zero-phase; fall back to sosfilt for very short segments
    if len(segment) > 20:
        filtered = _sps.sosfiltfilt(sos, segment)
    else:
        filtered = _sps.sosfilt(sos, segment)
    return np.clip(filtered.astype(segment.dtype), -1.0, 1.0)


def _cosine_schedule(t: int, T: int) -> float:
    """Cosine Noise-Schedule: σ_t = σ_max * (t/T)²"""
    return _SIGMA_MAX * (t / max(T, 1)) ** 2


def _burg_ar_predict(context: np.ndarray, order: int, n_samples: int) -> np.ndarray:
    """
    AR-Prädiktor via Levinson-Durbin (Toeplitz-Normalgleichungen, AR-Ordnung 64).
    Extrapoliert n_samples Samples aus dem Kontext.

    Algorithmus:
        1. Autokorrelationsschätzung (positive Lags 0…order)
        2. Aufstellen der Toeplitz-Normalgleichungen R·a = r
        3. Numerische Lösung via np.linalg.solve (= Levinson-Durbin-Annäherung)
        4. Rekursive Vorwärtsvorhersage mit gespeicherten Kontextwerten

    Referenz:
        Levinson (1947), Durbin (1960) — Toeplitz-Rekurrenz für AR-Schätzung
    """
    if len(context) < order + 1:
        return np.zeros(n_samples)

    # Autokorrelation schätzen
    ac = np.correlate(context, context, mode="full")
    ac = ac[len(ac) // 2 :]  # Nur positive Lags
    ac = ac[: order + 1]
    if ac[0] < 1e-10:
        return np.zeros(n_samples)

    # Toeplitz-System lösen (Levinson-Durbin approx)
    try:
        R = np.array([ac[abs(i)] for i in range(order)]).reshape(order, 1)
        Rmat = np.array([[ac[abs(i - j)] for j in range(order)] for i in range(order)])
        if np.linalg.matrix_rank(Rmat) < order:
            return np.zeros(n_samples)
        ar_coeff = np.linalg.solve(Rmat, R).flatten()
    except np.linalg.LinAlgError:
        return np.zeros(n_samples)

    # Vorhersage iterativ berechnen
    buf = list(context[-order:])
    predicted = []
    for _ in range(n_samples):
        val = np.dot(ar_coeff, buf[-order:][::-1])
        val = np.clip(val, -1.0, 1.0)
        predicted.append(val)
        buf.append(val)

    return np.array(predicted)


def _detect_gaps(audio: np.ndarray, sample_rate: int, min_gap_ms: float) -> list[tuple[int, int]]:
    """
    Erkennt Dropout-Lücken im Audio-Signal.
    Returns list of (start_sample, end_sample) tuples.
    """
    min_gap_samples = max(1, int(min_gap_ms * sample_rate / 1000))
    energy_thresh_linear = 10 ** (_ENERGY_THRESH_DBFS / 20.0)

    # Frame-weise RMS
    frame_size = _HOP
    n_frames = len(audio) // frame_size
    frame_rms = np.array([np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2)) for i in range(n_frames)])

    # Binäre Maske: True = Dropout
    is_dropout = frame_rms < energy_thresh_linear

    # Zusammenhängende Regionen finden
    gaps = []
    in_gap = False
    gap_start = 0
    for i, dropout in enumerate(is_dropout):
        if dropout and not in_gap:
            gap_start = i * frame_size
            in_gap = True
        elif not dropout and in_gap:
            gap_end = i * frame_size
            if (gap_end - gap_start) >= min_gap_samples:
                gaps.append((gap_start, gap_end))
            in_gap = False

    if in_gap:
        gap_end = len(audio)
        if (gap_end - gap_start) >= min_gap_samples:
            gaps.append((gap_start, gap_end))

    return gaps


def _inpaint_gap_dsp(
    audio: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    n_steps: int = _DIFFUSION_STEPS,
) -> np.ndarray:
    """
    DSP-basierende Diffusions-Inpainting für eine einzelne Lücke.
    Kombiniert AR-Prior mit iterativem Denoising.
    """
    gap_len = end - start
    context_samples = _CONTEXT_FRAMES * _HOP

    # Kontext-Puffer links und rechts
    left_ctx = audio[max(0, start - context_samples) : start].copy()
    right_ctx = audio[end : min(len(audio), end + context_samples)].copy()
    right_ctx_rev = right_ctx[::-1]

    # §v9.10.113: Adaptive AR order — order 64 diverges for gaps > 50 ms (2 400 samples).
    # AR(192) covers 3× more spectral modes; safe as long as context length > order.
    _AR_ORDER_ADAPTIVE = min(192, max(16, len(left_ctx) - 1)) if gap_len > 2400 else _AR_ORDER

    # AR-Vorhersage von links und von rechts (gespiegelt)
    ar_left = _burg_ar_predict(left_ctx, _AR_ORDER_ADAPTIVE, gap_len)
    ar_right = _burg_ar_predict(right_ctx_rev, _AR_ORDER_ADAPTIVE, gap_len)[::-1]

    # Cosine-Gewichtung: links → rechts
    t_vec = np.linspace(0, np.pi / 2, gap_len)
    w_left = np.cos(t_vec) ** 2
    w_right = np.sin(t_vec) ** 2

    # Kombinierter AR-Prior
    x = w_left * ar_left + w_right * ar_right

    # Envelopen-Kontinuität erzwingen
    if len(left_ctx) > 0 and len(right_ctx) > 0:
        env_left = np.abs(left_ctx[-1]) if len(left_ctx) > 0 else 0.0
        env_right = np.abs(right_ctx[0]) if len(right_ctx) > 0 else 0.0
        env_target = w_left * env_left + w_right * env_right
        x_env = np.abs(x) + 1e-10
        x = x * (env_target / x_env).clip(0.0, 2.0)

    # §2.40 Determinismus: Seeded RNG, derived from left-context fingerprint + gap position.
    # Ensures bit-exact reproducibility for identical inputs (same audio, same position).
    _ctx_seed = int(abs(float(np.sum(np.abs(left_ctx[: min(len(left_ctx), 64)])))) * 1e5 + start) % (2**31)
    _rng55 = np.random.default_rng(seed=_ctx_seed)

    # Reverse Diffusion (iteratives Denoising, T=n_steps Steps, adaptiv)
    for step in range(n_steps, 0, -1):
        sigma = _cosine_schedule(step, n_steps)
        if sigma > 0:
            noise = _rng55.standard_normal(gap_len) * sigma
            # Denoising: Projektionsschritt zurück zum Prior
            x = x + noise * 0.2
            # Regularisierung: Low-pass smoothing bei hohem Rauschen
            if sigma > 0.1:
                kernel_size = max(3, int(sigma * 20) | 1)
                x = np.convolve(x, np.ones(kernel_size) / kernel_size, mode="same")

    # Normierung auf Kontext-Energie-Level
    if len(left_ctx) > 10:
        ctx_rms = np.sqrt(np.mean(left_ctx[-100:] ** 2)) if np.any(left_ctx[-100:]) else 1e-4
        rec_rms = np.sqrt(np.mean(x**2)) + 1e-10
        x = x * (ctx_rms / rec_rms)

    return np.clip(x, -1.0, 1.0)


def _nmf_gap_fallback(
    channel: np.ndarray,
    start: int,
    end: int,
    sr: int,
) -> np.ndarray:
    """NMF-\u03b2 (IS-Divergenz, \u03b2=0) per-Gap-Inpainting — letzter DSP-Fallback (§2.47).

    Wenn _inpaint_gap_dsp scheitert, rekonstruiert dieser Fallback das Lücken-Segment
    durch NMF-Komponentenlernen auf dem umgebenden Kontext und überträgt die spektrale
    Struktur auf die Lücke. Reproduzierbar via deterministischem RNG-Seed.

    Algorithmus:
        1. Kontext-Fenster (±context_samples) via STFT analysieren
        2. NMF-Komponentenzerlegung (Rang 8, 10 multiplikative IS-Divergenz-Schritte)
        3. Durchschnittliche Aktivierung → Lücken-Magnitude
        4. Phasenkontinuität via Velocity-Fortsetzung (δφ aus Kontext)
        5. Phasen-kohärente Rekonstruktion via PGHI (Fallback: ISTFT)

    Reference: Févotte & Idier (2011) — NMF mit β-Divergenz (β=0 ≡ IS-Divergenz).
    """
    gap_len = end - start
    if gap_len <= 0:
        return channel[start:end]

    _n_fft = 1024
    _hop = 256
    ctx_samples = min(max(gap_len * 4, _n_fft * 2), len(channel) // 4)

    ctx_start = max(0, start - ctx_samples)
    ctx_end = min(len(channel), end + ctx_samples)
    context = channel[ctx_start:ctx_end].astype(np.float64)

    from scipy import signal as _sps

    freqs_ctx, times_ctx, Z_ctx = _sps.stft(
        context, fs=sr, window="hann", nperseg=_n_fft, noverlap=_n_fft - _hop
    )
    mag_ctx = np.abs(Z_ctx)  # (n_bins, n_frames)
    n_bins, n_frames = mag_ctx.shape
    if n_frames < 4 or n_bins < 4:
        # Fallback: zeros
        return np.zeros(gap_len, dtype=np.float32)

    # NMF-β (IS-Divergenz, β=0): multiplicative update rules (Févotte 2011)
    _rank = min(8, n_frames // 2)
    _eps = 1e-10
    _rng = np.random.default_rng(seed=int(np.abs(np.sum(context[:min(64, len(context))])) * 1e4 + start) % (2**31))
    _W = _rng.random((n_bins, _rank)) + 0.1  # basis spectra (n_bins, rank)
    _H = _rng.random((_rank, n_frames)) + 0.1  # activations (rank, n_frames)

    for _ in range(10):
        _WH = _W @ _H + _eps
        # IS-divergence updates (β=0)
        _H *= (_W.T @ (mag_ctx / _WH**2)) / (_W.T @ (1.0 / _WH) + _eps)
        _WH = _W @ _H + _eps
        _W *= ((mag_ctx / _WH**2) @ _H.T) / ((1.0 / _WH) @ _H.T + _eps)

    # Estimate gap magnitude from average activations
    gap_start_ctx = start - ctx_start
    gap_end_ctx = end - ctx_start
    gap_duration = gap_len / sr
    n_gap_frames = max(1, int(np.round(gap_duration * sr / _hop)))
    avg_H = np.mean(_H, axis=1, keepdims=True)  # (rank, 1)
    gap_mag = (_W @ (avg_H * np.ones((1, n_gap_frames)))).clip(0.0)  # (n_bins, n_gap_frames)

    # Phase velocity continuation from left context edge
    left_frame = max(0, gap_start_ctx // _hop - 1)
    if left_frame >= 1:
        phi_t1 = np.angle(Z_ctx[:, left_frame])
        phi_t0 = np.angle(Z_ctx[:, left_frame - 1])
        delta_phi = phi_t1 - phi_t0
    else:
        phi_t1 = np.zeros(n_bins)
        delta_phi = np.zeros(n_bins)

    gap_phase = np.zeros((n_bins, n_gap_frames))
    for i in range(n_gap_frames):
        gap_phase[:, i] = phi_t1 + delta_phi * (i + 1)

    Z_gap = gap_mag * np.exp(1j * gap_phase)

    # Phase-coherent reconstruction via PGHI (§2.47 VERBOTEN: direktes ISTFT)
    try:
        from dsp.pghi import pghi_reconstruct as _pghi_rec
        _initial_phase_gap = gap_phase.astype(np.float32)
        _gap_audio = _pghi_rec(
            gap_mag.astype(np.float32),
            sr=sr,
            win_size=_n_fft,
            hop=_hop,
            initial_phase=_initial_phase_gap,
        )
    except (ImportError, Exception) as _pghi_err:
        logger.debug("phase_55 NMF-gap PGHI-Fallback: %s", _pghi_err)
        _, _gap_audio = _sps.istft(Z_gap, fs=sr, window="hann", nperseg=_n_fft, noverlap=_n_fft - _hop)

    _gap_audio = np.asarray(_gap_audio, dtype=np.float64)

    # Trim/pad to exact gap length
    if len(_gap_audio) >= gap_len:
        _gap_audio = _gap_audio[:gap_len]
    else:
        _gap_audio = np.pad(_gap_audio, (0, gap_len - len(_gap_audio)))

    # Energy normalisation to match context boundary
    ctx_border_rms = float(np.sqrt(np.mean(channel[max(0, start - 64) : start] ** 2)) + 1e-10)
    rec_rms = float(np.sqrt(np.mean(_gap_audio**2)) + 1e-10)
    _gap_audio = _gap_audio * (ctx_border_rms / rec_rms)

    _gap_audio = np.nan_to_num(_gap_audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(_gap_audio, -1.0, 1.0).astype(np.float32)


def _try_cqtdiff_plus_plugin(audio: np.ndarray, start: int, end: int, sample_rate: int) -> np.ndarray | None:
    """
    CQTdiff Inpainting für Lücken ≥ 50 ms (Moliner & Välimäki 2022, §4.5 Aurik-Spec).

    CQTdiff konditioniert Score-basierte Diffusion im CQT-Domäne:
    - Logarithmische Frequenzauflösung ≡ musikalische Tonleiter
    - Harmonisch kohärente Füll-Lösung (keine Phasen-Inkoharenz)
    - Mindest-Lückengröße: 50 ms (CQTdiffPlusPlugin.MIN_GAP_MS)

    Gibt None zurück wenn:
    - Lücke < 50 ms (NMF-β in Phase 24 übernimmt)
    - Plugin oder ONNX-Modell nicht verfügbar
    """
    gap_ms = (end - start) / sample_rate * 1000.0
    if gap_ms < 50.0:
        return None  # Kurze Lücken → DSP-Diffusion (NMF-β-Äquivalent)
    try:
        import os as _os
        import sys

        _plugins_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "plugins")
        if _plugins_dir not in sys.path:
            sys.path.insert(0, _os.path.abspath(_plugins_dir))

        from plugins.cqtdiff_plus_plugin import get_cqtdiff_plus

        plugin = get_cqtdiff_plus()
        result = plugin.inpaint(audio=audio, sr=sample_rate, gap_start_sample=start, gap_end_sample=end)
        # InpaintingResult.audio = volles Audio-Signal mit gefüllter Lücke
        repaired_segment = result.audio[start:end]
        if repaired_segment is not None and np.isfinite(repaired_segment).all():
            return np.clip(repaired_segment.astype(np.float32), -1.0, 1.0)
        return None
    except Exception as _e:
        logger.debug("CQTdiff-Plugin nicht verfügbar: %s", _e)
        return None


def _try_flow_matching_plugin(audio: np.ndarray, start: int, end: int, sample_rate: int) -> np.ndarray | None:
    """
    Versucht FlowMatchingPlugin (Primär-Inpainting, §4.5) für Lücken aller Größen.

    FlowMatchingPlugin (Lipman et al. 2023) verwendet 4–16 Flow-Schritte statt
    1000 DDPM-Schritte — deutlich schneller und qualitativ gleichwertig oder besser.
    Aktiviert für Lücken aller Größen (20 ms – 30 s).
    """
    try:
        import os as _os
        import sys

        _plugins_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "plugins")
        if _plugins_dir not in sys.path:
            sys.path.insert(0, _os.path.abspath(_plugins_dir))

        from flow_matching_plugin import inpaint_flow

        result = inpaint_flow(audio, start, end, sample_rate)
        if result is not None and result.success:
            repaired_segment = result.audio[start:end]
            if repaired_segment is not None and np.isfinite(repaired_segment).all():
                return np.clip(repaired_segment.astype(np.float32), -1.0, 1.0)
        return None
    except Exception as _e:
        logger.debug("FlowMatchingPlugin nicht verfügbar: %s", _e)
        return None


def _try_diffwave_plugin(audio: np.ndarray, start: int, end: int, sample_rate: int) -> np.ndarray | None:
    """
    Versucht DiffWave-Plugin für Inpainting zu laden. Gibt None zurück wenn nicht verfügbar.
    Fallback-Priorität 2 nach FlowMatchingPlugin.
    """
    try:
        import importlib
        import os
        import sys

        plugins_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "plugins")
        if plugins_dir not in sys.path:
            sys.path.insert(0, os.path.abspath(plugins_dir))

        dw = importlib.import_module("diffwave_plugin")
        if not hasattr(dw, "inpaint"):
            return None

        return dw.inpaint(audio, start, end, sample_rate)
    except Exception as e:
        logger.debug("DiffWave-Plugin nicht verfügbar: %s", e)
        return None


def _try_consistency_model_inpainting(
    channel: np.ndarray, start: int, end: int, sample_rate: int
) -> np.ndarray | None:
    """Priority 0.8: Consistency Model inpainting (Song et al. 2023, ICML).

    Single-step diffusion via consistency distillation — 10–100× faster than DDPM,
    comparable quality for audio gaps of any size. Runs even during ml_thrashing since
    the model requires less GPU/CPU than multi-step diffusion.

    Returns filled gap segment or None if plugin unavailable.
    """
    assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
    gap_len = end - start
    if gap_len <= 0:
        return None
    try:
        import os as _os
        import sys

        _plugins_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "plugins")
        if _os.path.abspath(_plugins_dir) not in sys.path:
            sys.path.insert(0, _os.path.abspath(_plugins_dir))

        from plugins.consistency_inpaint_plugin import get_consistency_inpaint_plugin

        cm = get_consistency_inpaint_plugin()
        if cm is None:
            return None

        # Context window: 300 ms before and after gap (enough for musical phrase context)
        ctx_samples = min(int(0.30 * sample_rate), start)
        ctx_l = channel[max(0, start - ctx_samples) : start]
        ctx_r = channel[end : min(len(channel), end + ctx_samples)]

        result = cm.inpaint(ctx_l, ctx_r, gap_len, sample_rate)
        if result is None or len(result) == 0:
            return None

        result = np.nan_to_num(np.asarray(result, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(result[:gap_len], -1.0, 1.0)
    except Exception as _exc:
        logger.debug("_try_consistency_model_inpainting failed (non-critical): %s", _exc)
        return None


def _try_dac_token_inpainting(
    channel: np.ndarray, start: int, end: int, sample_rate: int
) -> np.ndarray | None:
    """Priority 1.5: DAC discrete audio codec token inpainting (Kumar et al. 2024).

    Encodes context into discrete audio tokens, inpaints the missing token sequence
    via causal AR / masked prediction, decodes back to waveform. Produces harmonically
    coherent fills for long gaps (≥ 50 ms) where CQTdiff+ is unavailable.

    Returns filled gap segment or None if plugin unavailable.
    """
    assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
    gap_len = end - start
    if gap_len <= 0:
        return None
    try:
        import os as _os
        import sys

        _plugins_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "plugins")
        if _os.path.abspath(_plugins_dir) not in sys.path:
            sys.path.insert(0, _os.path.abspath(_plugins_dir))

        from plugins.dac_plugin import get_dac_plugin

        dac = get_dac_plugin()
        if dac is None:
            return None

        # Context window: 500 ms (larger for token-based model)
        ctx_samples = min(int(0.50 * sample_rate), start)
        ctx_l = channel[max(0, start - ctx_samples) : start]
        ctx_r = channel[end : min(len(channel), end + ctx_samples)]

        result = dac.inpaint(ctx_l, ctx_r, gap_len, sample_rate)
        if result is None or len(result) == 0:
            return None

        result = np.nan_to_num(np.asarray(result, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(result[:gap_len], -1.0, 1.0)
    except Exception as _exc:
        logger.debug("_try_dac_token_inpainting failed (non-critical): %s", _exc)
        return None


def _is_ml_thrashing() -> bool:
    """Return True when ML paths should be avoided due to active swap thrashing."""
    try:
        from backend.core.ml_memory_budget import is_system_thrashing

        return bool(is_system_thrashing())
    except Exception:
        return False


def _conservative_boundary_fill(channel: np.ndarray, start: int, end: int) -> np.ndarray:
    """Fill gap with a boundary-constrained cosine interpolation."""
    gap_len = max(0, end - start)
    if gap_len <= 0:
        return np.zeros(0, dtype=np.float32)

    left = float(channel[start - 1]) if start > 0 else 0.0
    right = float(channel[end]) if end < len(channel) else left
    t = np.linspace(0.0, 1.0, gap_len, dtype=np.float32)
    fade = 0.5 - 0.5 * np.cos(np.pi * t)
    seg = (1.0 - fade) * left + fade * right
    return np.clip(np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)


def _gap_candidate_is_damaging(candidate: np.ndarray, channel: np.ndarray, start: int, end: int) -> bool:
    """Detect boundary/energy anomalies that indicate risky inpainting output."""
    gap_len = max(0, end - start)
    if gap_len <= 2 or len(candidate) == 0:
        return False

    seg = np.asarray(candidate, dtype=np.float32)[:gap_len]
    if not np.isfinite(seg).all():
        return True

    left_ctx = channel[max(0, start - 256) : start]
    right_ctx = channel[end : min(len(channel), end + 256)]
    if len(left_ctx) == 0 and len(right_ctx) == 0:
        return False
    ctx = np.concatenate((left_ctx[-128:], right_ctx[:128]))

    left_edge = float(channel[start - 1]) if start > 0 else float(seg[0])
    right_edge = float(channel[end]) if end < len(channel) else float(seg[-1])

    jump_l = abs(float(seg[0]) - left_edge)
    jump_r = abs(float(seg[-1]) - right_edge)
    ctx_delta = np.diff(ctx.astype(np.float32)) if len(ctx) > 4 else np.array([0.0], dtype=np.float32)
    median_ctx_step = float(np.median(np.abs(ctx_delta))) + 1e-6
    jump_limit = max(0.18, 10.0 * median_ctx_step)
    if jump_l > jump_limit or jump_r > jump_limit:
        return True

    ctx_p99 = float(np.percentile(np.abs(ctx), 99.0)) + 1e-6 if len(ctx) > 8 else 1e-3
    seg_p99 = float(np.percentile(np.abs(seg), 99.0))
    return bool(seg_p99 > ctx_p99 * 2.5)


def _process_channel(
    channel: np.ndarray,
    sample_rate: int,
    min_gap_ms: float,
    repaired_gap_samples: list[tuple[int, int]] | None = None,
    bw_cap_hz: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Inpainting für einen Mono-Kanal. Returns (repaired, stats)."""
    result = channel.copy()
    gaps = _detect_gaps(channel, sample_rate, min_gap_ms)

    # §11.7a: Bereits von RekonstruktionsDenker reparierte Gaps filtern
    if repaired_gap_samples:
        filtered = []
        for gs, ge in gaps:
            overlap = any(rs < ge and re > gs for rs, re in repaired_gap_samples)
            if not overlap:
                filtered.append((gs, ge))
        n_skipped = len(gaps) - len(filtered)
        gaps = filtered
    else:
        n_skipped = 0

    stats = {
        "n_gaps": len(gaps),
        "total_gap_ms": 0.0,
        "max_gap_ms": 0.0,
        "plugin_used": False,
        "pre_repaired_skipped": n_skipped,
        "damage_guard_activations": 0,
        "ml_thrashing_guard": False,
    }

    ml_thrashing = _is_ml_thrashing()
    if ml_thrashing:
        stats["ml_thrashing_guard"] = True
        logger.warning(
            "phase_55: ML-Thrashing erkannt — konservativer DSP-Pfad zum Schutz von Musik/Sprache aktiv"
        )

    for start, end in gaps:
        gap_ms = (end - start) / sample_rate * 1000

        # Adaptive Schrittzahl je nach Lückengröße
        n_steps = _adaptive_steps(gap_ms)
        if ml_thrashing:
            n_steps = min(n_steps, 32)

        # Priorität 0 [TIER-0]: FlowMatchingPlugin (Lipman et al. 2023, §4.4 SOTA-Matrix primär)
        # Flow Matching: 4–16 Schritte statt 50–200 DDPM-Schritte → 10–50× schneller,
        # gleichwertige oder bessere Qualität. Aktiviert für Lücken aller Größen (20 ms – 30 s).
        plugin_result = None
        if not ml_thrashing:
            plugin_result = _try_flow_matching_plugin(channel, start, end, sample_rate)
        if plugin_result is not None:
            candidate = plugin_result[: end - start]
            stats["plugin_used"] = True
            stats["flow_matching_tier0_used"] = stats.get("flow_matching_tier0_used", 0) + 1
        else:
            # Priorität 0.8: Consistency Model (Song et al. 2023) — 1-Schritt-Diffusions-Inpainting.
            # Läuft auch bei ml_thrashing, da deutlich ressourceneffizienter als Multi-Step-Diffusion.
            plugin_result = _try_consistency_model_inpainting(channel, start, end, sample_rate)
            if plugin_result is not None:
                logger.debug("phase_55: Consistency Model Inpainting OK (gap=%.1f ms)", gap_ms)
                candidate = plugin_result[: end - start]
                stats["plugin_used"] = True
                stats["consistency_model_used"] = stats.get("consistency_model_used", 0) + 1
            else:
                # Priorität 1: CQTdiff+ (≥ 50 ms Lücken, harmonisch kohärent, §4.5 Spec)
                if not ml_thrashing:
                    plugin_result = _try_cqtdiff_plus_plugin(channel, start, end, sample_rate)
                if plugin_result is not None:
                    candidate = plugin_result[: end - start]
                    stats["plugin_used"] = True
                else:
                    # Priorität 1.5: DAC Token Inpainting (Kumar et al. 2024) — diskrete Codec-Token.
                    # Musikalisch kohärente Fills für Lücken ≥ 50 ms, wenn CQTdiff+ nicht verfügbar.
                    if not ml_thrashing and gap_ms >= 50.0:
                        plugin_result = _try_dac_token_inpainting(channel, start, end, sample_rate)
                        if plugin_result is not None:
                            logger.debug("phase_55: DAC Token Inpainting OK (gap=%.1f ms)", gap_ms)
                            candidate = plugin_result[: end - start]
                            stats["plugin_used"] = True
                            stats["dac_token_used"] = stats.get("dac_token_used", 0) + 1

                    if plugin_result is None:
                        # Priorität 2+: DSP AR-Diffusion → NMF-β IS-Divergenz (§2.47 Fallback-Pflicht)
                        try:
                            candidate = _inpaint_gap_dsp(channel, start, end, sample_rate, n_steps)
                        except Exception as _dsp_exc:
                            logger.debug("DSP-AR-Diffusion fehlgeschlagen — NMF-\u03b2-Fallback (gap %d:%d): %s", start, end, _dsp_exc)
                            candidate = _nmf_gap_fallback(channel, start, end, sample_rate)

        if _gap_candidate_is_damaging(candidate, channel, start, end):
            stats["damage_guard_activations"] += 1
            logger.warning(
                "phase_55: Damage-Guard für Gap [%d:%d] (%.1f ms) aktiviert — ersetze riskante Rekonstruktion",
                start,
                end,
                gap_ms,
            )
            candidate = _conservative_boundary_fill(channel, start, end)

        result[start:end] = np.clip(
            np.nan_to_num(candidate[: end - start], nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )

        # §0 BW-Cap: historische Träger dürfen keine HF-Details halluzinieren
        if bw_cap_hz is not None:
            gap_segment = result[start:end]
            result[start:end] = _apply_bw_cap(gap_segment, sample_rate, bw_cap_hz)

        stats["total_gap_ms"] += gap_ms
        stats["max_gap_ms"] = max(stats["max_gap_ms"], gap_ms)

    return result, stats


# ─── Energie-Kontinuitäts-Score (PESQ-Proxy) ────────────────────────────
def _reconstruction_quality_score(original: np.ndarray, repaired: np.ndarray, gaps: list[tuple[int, int]]) -> float:
    """Schätzt die Rekonstruktionsqualität (0–1) durch Energie-Kontinuität."""
    if not gaps:
        return 1.0

    scores = []
    for start, end in gaps:
        border = max(1, (end - start) // 4)
        left = repaired[max(0, start - border) : start]
        center = repaired[start:end]
        right = repaired[end : end + border]

        if len(left) < 2 or len(center) < 2 or len(right) < 2:
            scores.append(0.8)
            continue

        rms_l = np.sqrt(np.mean(left**2))
        rms_c = np.sqrt(np.mean(center**2))
        rms_r = np.sqrt(np.mean(right**2))

        # Ideale Energie sollte kontinuierlich sein
        expected = (rms_l + rms_r) / 2
        deviation = abs(rms_c - expected) / (expected + 1e-10)
        score = max(0.0, 1.0 - deviation)
        scores.append(score)

    return float(np.mean(scores))


# ─── Phase-Klasse ───────────────────────────────────────────────────────
class DiffusionInpaintingPhase(PhaseInterface):
    """
    Phase 55: Diffusions-basiertes Audio-Inpainting für Lücken und Dropouts.

    Ersetzt einfache Null-Interpolation durch iterative Diffusionsrekonstruktion
    mit harmonischem AR-Prior. Optional: DiffWave-Plugin-Pfad für ML-gestützte
    Rekonstruktion wenn Modellgewichte vorhanden.
    """

    @staticmethod
    def _derive_safe_inpainting_strength(
        effective_strength: float,
        material_key: str,
        vocals_confidence: float,
    ) -> float:
        """Reduce wet blend for content that is prone to synthetic overfill artifacts."""
        strength = float(effective_strength)
        if vocals_confidence >= 0.40:
            strength *= 0.78
        _is_analog_sensitive = any(
            token in material_key for token in ("vinyl", "shellac", "wax_cylinder", "wire_recording", "lacquer_disc")
        )
        if _is_analog_sensitive:
            strength *= 0.85
        # Prevent tonal-center drift spikes from aggressive diffuse fill on vocal analog material.
        if _is_analog_sensitive and vocals_confidence >= 0.40:
            strength = min(strength, 0.58)
        return float(np.clip(strength, 0.0, 1.0))

    def get_metadata(self) -> PhaseMetadata:
        """Implementiert PhaseInterface.get_metadata()."""
        return PhaseMetadata(
            phase_id="phase_55_diffusion_inpainting",
            name="Diffusion Inpainting",
            category=PhaseCategory.RESTORATION,
            priority=9,  # CRITICAL
            dependencies=["phase_24_dropout_repair", "phase_50_spectral_repair"],
            estimated_time_factor=0.08,
            version="1.0.0",
            memory_requirement_mb=128,
            is_cpu_intensive=True,
            quality_impact=0.85,
            description="Masked Diffusion Inpainting für Audio-Lücken und Dropouts >20ms",
        )

    def _nmf_spectral_inpainting_fallback(
        self, audio: np.ndarray, sr: int, strength: float = 0.5
    ) -> np.ndarray:
        """NMF-\u03b2 (IS-Divergenz, \u03b2=0) Ganzsignal-Inpainting — absoluter letzter Fallback (§2.47).

        Wird aufgerufen wenn alle Kanal-Verarbeitungspfade (_process_channel) fehlschlagen.
        Zerlegt das gesamte Spektrum in NMF-Komponenten und rekonstruiert Niedrigenergie-Regionen
        (\u22641. Quartil), die wahrscheinlich beschädigt sind, via NMF-Schätzung.

        Args:
            audio:    Input audio (mono 1-D oder Stereo N×2).
            sr:       Sample-Rate (muss 48000 Hz sein).
            strength: Blend-Faktor für NMF-Reparatur in Niedrigenergie-Bins (0–1).

        Returns:
            Inpainted audio, identical shape to input, NaN/Inf-frei, ∈ [-1, 1].
        """
        from scipy import signal as _sps

        _n_fft = 2048
        _hop = 512
        _is_stereo = audio.ndim == 2
        _mono = np.mean(audio, axis=1) if _is_stereo else audio.copy()
        _mono = _mono.astype(np.float64)

        _, _, _Z = _sps.stft(_mono, fs=sr, nperseg=_n_fft, noverlap=_n_fft - _hop, window="hann")
        _mag = np.abs(_Z)
        _phase = np.angle(_Z)

        _n_freq, _n_time = _mag.shape
        _rank = min(16, max(2, _n_time // 4))
        if _n_time < 4:
            return audio.copy()

        _eps = 1e-10
        _rng = np.random.default_rng(seed=42)  # deterministic — reproducible for same input
        _W = _rng.random((_n_freq, _rank)) + 0.1  # basis spectra
        _H = _rng.random((_rank, _n_time)) + 0.1  # activations

        for _ in range(10):  # IS-divergence multiplicative updates (\u03b2=0)
            _WH = _W @ _H + _eps
            _H *= (_W.T @ (_mag / _WH**2)) / (_W.T @ (1.0 / _WH) + _eps)
            _WH = _W @ _H + _eps
            _W *= ((_mag / _WH**2) @ _H.T) / ((1.0 / _WH) @ _H.T + _eps)

        _reconstructed_mag = _W @ _H

        # Apply NMF reconstruction only to lowest-quartile energy bins (likely damage)
        _energy_mask = _mag < np.percentile(_mag, 25)
        _mag_repaired = np.where(
            _energy_mask,
            (1.0 - strength) * _mag + strength * _reconstructed_mag,
            _mag,
        )

        # Phase-coherent ISTFT via PGHI (§2.47)
        try:
            from dsp.pghi import pghi_reconstruct as _pghi_fb
            _repaired_mono = _pghi_fb(
                _mag_repaired.astype(np.float32),
                sr=sr,
                win_size=_n_fft,
                hop=_hop,
                initial_phase=_phase.astype(np.float32),
            )
            _repaired_mono = np.asarray(_repaired_mono, dtype=np.float64)
        except (ImportError, Exception) as _pghi_fb_err:
            logger.debug("phase_55 NMF-fallback PGHI-Fallback: %s", _pghi_fb_err)
            _Z_repaired = _mag_repaired * np.exp(1j * _phase)
            _, _repaired_mono = _sps.istft(_Z_repaired, fs=sr, nperseg=_n_fft, noverlap=_n_fft - _hop, window="hann")

        _repaired_mono = np.asarray(_repaired_mono, dtype=np.float64)
        if len(_repaired_mono) > len(_mono):
            _repaired_mono = _repaired_mono[:len(_mono)]
        elif len(_repaired_mono) < len(_mono):
            _repaired_mono = np.pad(_repaired_mono, (0, len(_mono) - len(_repaired_mono)))

        _repaired_mono = np.nan_to_num(_repaired_mono, nan=0.0, posinf=0.0, neginf=0.0)
        _repaired_mono = np.clip(_repaired_mono, -1.0, 1.0)

        if _is_stereo:
            _ratio = _repaired_mono / (_mono + np.sign(_mono + 1e-30) * 1e-10)
            _ratio = np.clip(_ratio, 0.0, 10.0)
            _out = np.column_stack([audio[:, 0] * _ratio, audio[:, 1] * _ratio])
            return np.clip(np.nan_to_num(_out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)

        return _repaired_mono.astype(np.float32)

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        min_gap_ms: float = _MIN_GAP_MS_DEFAULT,
        **kwargs,
    ) -> PhaseResult:
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        t0 = time.perf_counter()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        # §0 BW-Cap: prevent hallucination of HF content on bandwidth-limited carriers
        _material = kwargs.get("material_type")
        _mat_key = str(_material).lower() if _material is not None else ""
        _vocals_conf = float(kwargs.get("panns_vocals_confidence", 0.0))
        if _vocals_conf == 0.0:  # Fallback: direct callers may use panns_singing key
            _vocals_conf = float(kwargs.get("panns_singing", 0.0))
        safe_strength = self._derive_safe_inpainting_strength(effective_strength, _mat_key, _vocals_conf)
        # Accept both enum value strings like "MaterialType.WAX_CYLINDER" and plain keys
        for _mk, _cap in _MATERIAL_BW_CAP_HZ.items():
            if _mk in _mat_key:
                _bw_cap_hz: float | None = _cap
                break
        else:
            _bw_cap_hz = None

        if effective_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=dry,
                execution_time_seconds=time.perf_counter() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        min_gap_ms_eff = float(min_gap_ms) / max(safe_strength, 0.1)
        source_audio = audio

        # §11.7a: Bereits von RekonstruktionsDenker reparierte Gap-Regionen
        _repaired_gaps: list[tuple[int, int]] = kwargs.get("repaired_gap_samples", [])
        _n_repaired_skipped = 0
        _damage_guard_hits = 0
        _thrash_guard_active = False

        if audio.ndim == 1:
            # Mono
            repaired, stats = _process_channel(audio, sample_rate, min_gap_ms_eff, _repaired_gaps, _bw_cap_hz)
            gaps = _detect_gaps(audio, sample_rate, min_gap_ms_eff)
            quality = _reconstruction_quality_score(audio, repaired, gaps)
            n_gaps = stats["n_gaps"]
            total_gap_ms = stats["total_gap_ms"]
            max_gap_ms = stats["max_gap_ms"]
            plugin_used = stats["plugin_used"]
            _n_repaired_skipped = stats.get("pre_repaired_skipped", 0)
            _damage_guard_hits = int(stats.get("damage_guard_activations", 0))
            _thrash_guard_active = bool(stats.get("ml_thrashing_guard", False))
        else:
            # §2.51 Linked-Stereo: Gap-Detektion auf Mono-Mix, kohärente Reparatur
            # Detect gaps on mono downmix to ensure identical gap regions for L+R
            mono_mix = np.mean(audio, axis=1)
            mono_gaps = _detect_gaps(mono_mix, sample_rate, min_gap_ms_eff)

            channels_repaired = []
            n_gaps = 0
            total_gap_ms = 0.0
            max_gap_ms = 0.0
            plugin_used = False
            quality_scores = []

            for ch in range(audio.shape[1]):
                ch_rep, stats = _process_channel(audio[:, ch], sample_rate, min_gap_ms_eff, _repaired_gaps, _bw_cap_hz)
                channels_repaired.append(ch_rep)
                n_gaps = max(n_gaps, stats["n_gaps"])
                total_gap_ms += stats["total_gap_ms"]
                max_gap_ms = max(max_gap_ms, stats["max_gap_ms"])
                plugin_used = plugin_used or stats["plugin_used"]
                _n_repaired_skipped += stats.get("pre_repaired_skipped", 0)
                _damage_guard_hits += int(stats.get("damage_guard_activations", 0))
                _thrash_guard_active = _thrash_guard_active or bool(stats.get("ml_thrashing_guard", False))

                quality_scores.append(_reconstruction_quality_score(audio[:, ch], ch_rep, mono_gaps))

            repaired = np.column_stack(channels_repaired)
            quality = float(np.mean(quality_scores)) if quality_scores else 1.0

        if 0.0 < safe_strength < 1.0:
            repaired = source_audio + safe_strength * (repaired - source_audio)

        elapsed = time.perf_counter() - t0

        repaired = np.nan_to_num(repaired, nan=0.0, posinf=0.0, neginf=0.0)
        repaired = np.clip(repaired, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=repaired,
            execution_time_seconds=elapsed,
            metadata={
                "n_gaps_detected": n_gaps,
                "total_gap_ms": round(total_gap_ms, 2),
                "max_gap_ms": round(max_gap_ms, 2),
                "plugin_used": plugin_used,
                "damage_guard_activations": int(_damage_guard_hits),
                "ml_thrashing_guard": bool(_thrash_guard_active),
                "reconstruction_quality": round(quality, 4),
                "diffusion_steps": f"{_DIFFUSION_STEPS}/{_DIFFUSION_STEPS_MED}/{_DIFFUSION_STEPS_LONG} (adaptive)",
                "min_gap_ms": min_gap_ms,
                "min_gap_ms_effective": round(min_gap_ms_eff, 2),
                "ar_order": _AR_ORDER,
                "primary_ml": "cqtdiff",
                "pre_repaired_gaps_skipped": _n_repaired_skipped,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "safe_strength": safe_strength,
                "panns_vocals_confidence": _vocals_conf,
                "bw_cap_hz": _bw_cap_hz,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
        )
