import numpy as np


def safe_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono, handling both (N, 2) and (2, N) layouts safely.

    Respects §2.51 Stereo-Kohärenz-Invariante: Convert to mono without loss
    of phase information or spectral coherence.

    Args:
        audio: Input audio, 1D (mono) or 2D (stereo in any orientation)

    Returns:
        Mono audio as 1D numpy array (or scalar for degenerate inputs)
    """
    if audio.ndim == 1:
        return audio

    # Ensure float64 for precision
    audio = audio.astype(np.float64)

    # Determine orientation and convert safely
    if audio.shape[0] == 2 and audio.shape[1] > 2:
        # (2, N) channels-first → mean over channels (axis=0)
        return np.mean(audio, axis=0)
    elif audio.shape[0] == 2 and audio.shape[1] == 2:
        # Edge case: exactly (2, 2) — ambiguous, but treat as (2, N) channels-first
        # This gives a (2,) output
        return np.mean(audio, axis=0)
    elif audio.shape[1] == 2:
        # (N, 2) channels-last → mean over channels (axis=1)
        return np.mean(audio, axis=1)
    else:
        # Ambiguous: use heuristic based on which dimension is smaller
        # (channels are typically 2, samples >> 2)
        axis = 0 if audio.shape[0] < audio.shape[1] else 1
        return np.mean(audio, axis=axis)


def stereo_channel_view(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stereo channels as 1D arrays for either (2, N) or (N, 2) layout."""
    if audio.ndim != 2:
        raise ValueError(f"Stereo audio must be 2D, got shape {audio.shape}")
    if audio.shape[0] == 2 and audio.shape[1] > 2:
        return audio[0], audio[1]
    if audio.shape[1] == 2:
        return audio[:, 0], audio[:, 1]
    if audio.shape[0] == 2 and audio.shape[1] == 2:
        return audio[0], audio[1]
    raise ValueError(f"Unsupported stereo layout: {audio.shape}")


def stereo_like(left: np.ndarray, right: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Rebuild stereo audio while preserving the template orientation."""
    if template.ndim != 2:
        raise ValueError(f"Stereo template must be 2D, got shape {template.shape}")
    if template.shape[0] == 2 and template.shape[1] > 2:
        return np.vstack([left, right])
    if template.shape[1] == 2:
        return np.column_stack([left, right])
    if template.shape[0] == 2 and template.shape[1] == 2:
        return np.vstack([left, right])
    raise ValueError(f"Unsupported stereo template layout: {template.shape}")


def to_channels_last(audio: np.ndarray) -> tuple["np.ndarray", bool]:
    """Normalize stereo audio to (N, 2) channels-last layout.

    Returns (normalized_audio, was_transposed) so the caller can restore the
    original orientation with ``restore_layout``.
    """
    if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
        return audio.T, True
    return audio, False


def restore_layout(audio: np.ndarray, was_transposed: bool) -> np.ndarray:
    """Undo a ``to_channels_last`` transposition if it was applied."""
    if was_transposed and audio.ndim == 2:
        return audio.T
    return audio


def audio_sample_count(audio: np.ndarray) -> int:
    """Return the time-axis sample count for mono or stereo audio."""
    if audio.ndim == 1:
        return int(audio.shape[0])
    if audio.ndim == 2:
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            return int(audio.shape[1])
        return int(audio.shape[0])
    raise ValueError(f"Unsupported audio rank for sample count: {audio.shape}")


def compute_gated_rms_linear(sig: np.ndarray, gate_dbfs: float = -50.0) -> float:
    """Compute frame-gated RMS in linear scale (stereo-safe via mono energy).

    §2.45a v9.12.1: Adaptive gate (same as _rms_dbfs_gated in UV3).
    effective_gate = max(gate_dbfs, P5+10) — excludes vinyl/shellac surface-noise
    frames (-35 to -45 dBFS) so that noise removal is not misread as a music-level drop.
    Old fixed gate (-50 dBFS) included all noise frames → false drop → Pegelexplosion.
    """
    x = np.asarray(sig, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if x.ndim == 2:
        if x.shape[0] <= 2 and x.shape[1] > x.shape[0]:
            x = np.mean(x, axis=0)
        else:
            x = np.mean(x, axis=1)
    frame = 480
    n = int(x.shape[0])
    if n < frame:
        return float(np.sqrt(np.mean(x * x)) + 1e-12)

    # Collect all frame energies first (for adaptive gate computation)
    all_frame_power: list[float] = []
    for i in range(0, n - frame + 1, frame):
        f = x[i : i + frame]
        all_frame_power.append(float(np.mean(f * f)))

    # §2.45a Adaptive gate: P5+10 dB above noise floor
    effective_gate_dbfs = gate_dbfs
    if len(all_frame_power) >= 10:
        _p5_power = float(np.percentile(all_frame_power, 5))
        if _p5_power > 0.0:
            _p5_db = 10.0 * float(np.log10(_p5_power + 1e-12))
            _adaptive = _p5_db + 10.0
            if _adaptive > gate_dbfs:  # True whenever P5 > gate-10 (all real audio)
                effective_gate_dbfs = min(_adaptive, gate_dbfs + 25.0)

    gate_lin2 = 10.0 ** (effective_gate_dbfs / 10.0)
    vals: list[float] = [p for p in all_frame_power if p > gate_lin2]
    if not vals:
        return float(np.sqrt(np.mean(x * x)) + 1e-12)
    return float(np.sqrt(float(np.mean(vals))) + 1e-12)


def compute_gated_rms_dbfs(sig: np.ndarray, gate_dbfs: float = -50.0) -> float:
    """Compute frame-gated RMS in dBFS."""
    rms = compute_gated_rms_linear(sig, gate_dbfs=gate_dbfs)
    return float(20.0 * np.log10(rms + 1e-12))


def apply_musical_gain_envelope(
    audio: np.ndarray,
    gain: float,
    gate_dbfs: float = -36.0,
    crossfade_ms: float = 10.0,
    sr: int = 48000,
    reference_for_gate: np.ndarray | None = None,
) -> np.ndarray:
    """§2.45a-II: Apply makeup gain ONLY to musical frames, leaving silence at gain=1.0.

    Silence frames (frame RMS <= effective_gate_dbfs) remain at unity gain.
    A short crossfade (box-blur on the gate envelope) prevents hard clicks at
    music/silence boundaries.

    Adaptive gate (§2.45a noise-floor-aware, v9.12.1):
        effective_gate = max(gate_dbfs, P5+10) where P5 is the 5th-percentile frame RMS.

        When ``reference_for_gate`` is provided (e.g. the pre-phase audio), P5 is
        computed from that signal instead of from ``audio``.  This is critical for
        partially-denoised audio: after phase_28/phase_09 remove surface noise, some
        frames drop to -55 dBFS (fully denoised) dragging P5 down to -50 dBFS.
        With P5=-50: P5+10=-40, condition -40>-36.0 is False → gate stays at -36 →
        residual noise at -35 dBFS gets amplified → Pegelexplosion.
        Using the pre-phase reference (P5≈-42 for vinyl): P5+10=-32>-36 → gate=-32 →
        residual noise at -35 excluded → no amplification.

    Args:
        audio:              Input audio (1D or 2D float32).
        gain:               Linear gain factor (>= 1.0; values <= 1.0005 are skipped).
        gate_dbfs:          Nominal frame energy threshold below which a frame is silence.
        crossfade_ms:       Width of the smoothing window at music/silence transitions.
        sr:                 Sample rate used to convert crossfade_ms to samples.
        reference_for_gate: Optional pre-phase audio for adaptive-gate P5 estimation.
                            When set, the gate is computed from this signal's noise floor
                            (not from the post-processing audio's altered P5).

    Returns:
        Audio with gain applied only on musical frames, same shape and dtype.
    """
    if gain <= 1.0005:
        return audio
    arr = np.asarray(audio, dtype=np.float32)
    was_2d = arr.ndim == 2
    # Build mono energy signal for gate detection (from audio being amplified)
    if was_2d:
        ch_first = arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]
        mono = np.mean(arr, axis=0) if ch_first else np.mean(arr, axis=1)
    else:
        mono = arr
    n = len(mono)
    frame_len = 480  # 10 ms @ 48 kHz
    n_full = max(1, n // frame_len)

    # --- Pass 1: collect per-frame RMS values (for gate-envelope construction) ---
    frame_rms_db: list[float] = []
    for fi in range(n_full):
        s = fi * frame_len
        e = min(s + frame_len, n)
        chunk = mono[s:e].astype(np.float64)
        frame_rms_db.append(float(20.0 * np.log10(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)) + 1e-12)))
    tail_rms_db: float | None = None
    tail_s = n_full * frame_len
    if tail_s < n:
        tail = mono[tail_s:].astype(np.float64)
        tail_rms_db = float(20.0 * np.log10(float(np.sqrt(np.mean(tail * tail) + 1e-12)) + 1e-12))
        frame_rms_db.append(tail_rms_db)

    # --- Adaptive gate: raise threshold to exclude surface-noise frames ---
    # §2.45a v9.12.1: effective_gate = max(gate_dbfs, P5+10).
    # P5 is computed from ``reference_for_gate`` (pre-phase audio) when provided:
    # After partial denoising, the processed audio's P5 drops to -50+ dBFS (fully-
    # denoised frames drag it down), so the gate logic based on the processed audio
    # fails to exclude residual vinyl-noise frames at -35 dBFS → Pegelexplosion.
    # Using the PRE-phase reference preserves the correct noise-floor estimate.
    _gate_source_rms_db: list[float]
    if reference_for_gate is not None:
        try:
            _ref_arr = np.asarray(reference_for_gate, dtype=np.float32)
            if _ref_arr.ndim == 2:
                _ref_ch = _ref_arr.shape[0] <= 2 and _ref_arr.shape[1] > _ref_arr.shape[0]
                _ref_mono = np.mean(_ref_arr, axis=0) if _ref_ch else np.mean(_ref_arr, axis=1)
            else:
                _ref_mono = _ref_arr
            _ref_n = len(_ref_mono)
            _ref_n_full = max(1, _ref_n // frame_len)
            _gate_source_rms_db = []
            for _fi in range(_ref_n_full):
                _s, _e = _fi * frame_len, min((_fi + 1) * frame_len, _ref_n)
                _c = _ref_mono[_s:_e].astype(np.float64)
                _gate_source_rms_db.append(float(20.0 * np.log10(float(np.sqrt(np.mean(_c * _c) + 1e-12)) + 1e-12)))
            if _ref_n > _ref_n_full * frame_len:
                _tail_c = _ref_mono[_ref_n_full * frame_len :].astype(np.float64)
                _gate_source_rms_db.append(
                    float(20.0 * np.log10(float(np.sqrt(np.mean(_tail_c * _tail_c) + 1e-12)) + 1e-12))
                )
        except Exception:
            _gate_source_rms_db = frame_rms_db
    else:
        _gate_source_rms_db = frame_rms_db

    effective_gate = gate_dbfs
    if len(_gate_source_rms_db) >= 10:
        p5_rms_db = float(np.percentile(_gate_source_rms_db, 5))
        _adaptive = p5_rms_db + 10.0
        if _adaptive > gate_dbfs:
            effective_gate = min(_adaptive, gate_dbfs + 25.0)

    # --- Pass 2: build gate envelope using adaptive threshold ---
    gate_env = np.zeros(n, dtype=np.float32)
    full_rms = frame_rms_db[:n_full] if tail_rms_db is not None else frame_rms_db
    for fi, rms_db in enumerate(full_rms):
        if rms_db > effective_gate:
            s = fi * frame_len
            e = min(s + frame_len, n)
            gate_env[s:e] = 1.0
    if tail_rms_db is not None and tail_rms_db > effective_gate:
        gate_env[tail_s:] = 1.0

    # Smooth transitions
    cf_samples = max(1, int(crossfade_ms * sr / 1000.0))
    if cf_samples > 1:
        kernel = np.ones(cf_samples, dtype=np.float32) / cf_samples
        gate_env = np.convolve(gate_env, kernel, mode="same")
        gate_env = np.clip(gate_env, 0.0, 1.0)
    per_sample_gain = (1.0 + (gain - 1.0) * gate_env).astype(np.float32)

    # §2.30b Stufe 5 — Per-sample quiet-zone hard clamp (AFTER smoothing)
    # Box-blur/crossfade smoothing bleeds positive gain into frames bordering
    # quiet zones (fadeout, intro hiss).  Any 10 ms frame whose pre-smoothing
    # RMS was ≤ effective_gate MUST NOT receive a per-sample gain > 1.0,
    # regardless of what the smoothed gate_env says.
    # v9.12.1: Use effective_gate (adaptive) instead of hardcoded -36 dBFS.
    # For real vinyl: effective_gate ≈ -28 dBFS → vinyl surface noise (-35 dBFS)
    # is correctly clamped.  With -36 dBFS, noise at -35 dBFS was NOT clamped
    # (−35 > −36) → Pegelexplosion after boundary crossfade.
    # Floor at gate_dbfs (−36) so we never clamp more aggressively than -36 on
    # non-vinyl material where effective_gate might be lower than gate_dbfs.
    _QUIET_ZONE_DB = max(gate_dbfs, effective_gate)
    for _fi, _rdb in enumerate(full_rms):
        if _rdb <= _QUIET_ZONE_DB:
            _fs = _fi * frame_len
            _fe = min(_fs + frame_len, n)
            per_sample_gain[_fs:_fe] = np.minimum(per_sample_gain[_fs:_fe], 1.0)
    if tail_rms_db is not None and tail_rms_db <= _QUIET_ZONE_DB:
        per_sample_gain[tail_s:] = np.minimum(per_sample_gain[tail_s:], 1.0)

    if was_2d:
        ch_first = arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]
        if ch_first:
            return (arr * per_sample_gain[np.newaxis, :]).astype(np.float32)
        return (arr * per_sample_gain[:, np.newaxis]).astype(np.float32)
    return (arr * per_sample_gain).astype(np.float32)


def check_gain_safety(
    audio: np.ndarray,
    requested_gain: float,
    max_peak_dbfs: float = -1.0,
) -> tuple[float, bool]:
    """Pre-flight: compute the maximum gain that won't clip the audio.

    §2.51a / §2.45a preventive approach: calculate max safe gain BEFORE
    applying it, so Pegelexplosion can never happen in the first place.

    Uses 99.9th-percentile peak (§DSP-invariant) to avoid impulse artefacts
    (crackle, clicks) blocking normalisation of the musical content.

    Args:
        audio:          Input audio (any shape float32).
        requested_gain: Desired linear gain factor.
        max_peak_dbfs:  Hard ceiling in dBFS (default -1.0 dBTP, broadcast-safe).

    Returns:
        (safe_gain, was_clamped) where safe_gain ≤ requested_gain and
        was_clamped=True iff the gain was reduced to stay under the ceiling.
    """
    if requested_gain <= 1.0005:
        return float(requested_gain), False
    arr = np.asarray(audio, dtype=np.float32)
    peak99 = float(np.percentile(np.abs(arr), 99.9))
    if peak99 < 1e-9:
        return 1.0, True  # Silent — no positive gain allowed
    max_peak_linear = float(10.0 ** (max_peak_dbfs / 20.0))
    max_safe = max_peak_linear / peak99
    if requested_gain <= max_safe:
        return float(requested_gain), False
    return float(max(1.0, max_safe)), True
