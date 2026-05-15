from typing import cast

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
        return np.asarray(np.mean(audio, axis=0))
    if audio.shape[0] == 2 and audio.shape[1] == 2:
        # Edge case: exactly (2, 2) — ambiguous, but treat as (2, N) channels-first
        # This gives a (2,) output
        return np.asarray(np.mean(audio, axis=0))
    if audio.shape[1] == 2:
        # (N, 2) channels-last → mean over channels (axis=1)
        return np.asarray(np.mean(audio, axis=1))
    # Ambiguous: use heuristic based on which dimension is smaller
    # (channels are typically 2, samples >> 2)
    axis = 0 if audio.shape[0] < audio.shape[1] else 1
    return np.asarray(np.mean(audio, axis=axis))


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


# §2.45a: Per-material noise floor gate used as hard minimum in compute_signal_relative_gate_dbfs.
# Values = typical noise floor + 6 dB margin (AES/iZotope RX practice).
# Vinyl ≈ -33 dBFS noise → gate -27 dBFS; shellac ≈ -20 dBFS → gate -14 dBFS.
_MATERIAL_GATE_DBFS: dict[str, float] = {
    "shellac": -14.0,
    "wax_cylinder": -10.0,
    "lacquer_disc": -20.0,
    "wire_recording": -20.0,
    "acoustic_78": -14.0,
    "vinyl": -27.0,
    "reel_tape": -32.0,
    "cassette": -38.0,
    "tape": -32.0,
    "mp3_low": -44.0,
    "mp3_medium": -46.0,
    "cd_digital": -48.0,
    "streaming": -48.0,
    "dat": -48.0,
    "minidisc": -44.0,
    "unknown": -36.0,
}


def compute_signal_relative_gate_dbfs(  # pylint: disable=too-many-positional-arguments
    reference_audio: np.ndarray,
    margin_db: float = 9.0,
    percentile: float = 15.0,
    fallback_gate_dbfs: float = -36.0,
    frame_len: int = 480,
    material_key: str | None = None,
) -> float:
    """§2.45a Material-adaptive gate: signal-relative threshold (CEDAR/iZotope RX approach).

    Professional tools (CEDAR, iZotope RX 11, Waves Z-Noise) measure the noise floor of
    the actual source signal and set the gate = noise_floor + margin (6–10 dB).
    This avoids the failure mode of fixed absolute thresholds (e.g. -36.0 dBFS) when
    the source noise floor is higher than the threshold (vinyl: -33 dBFS > -36 dBFS).

    Uses the P15 percentile of frame RMS values (not P5) to get a robust noise floor
    estimate that stays in the actual noise region even for loud pop/rock content where
    P5 can fall into the music region.

    The computed gate can only be equal to or HIGHER than the material floor from
    _MATERIAL_GATE_DBFS — the material floor acts as a hard minimum (same design as
    CEDAR minimum-statistics: measured floor + margin, bounded by known carrier floor).

    Args:
        reference_audio: Pre-phase source audio (the signal whose noise floor to estimate).
        margin_db:        dB margin above noise floor (default 9 dB, AES/iZotope practice).
        percentile:       Percentile of frame RMS to use as noise floor (default 15).
        fallback_gate_dbfs: Used when reference is too short to estimate (< 10 frames).
        frame_len:        Frame length in samples (default 480 = 10 ms @ 48 kHz).
        material_key:     Optional material type (e.g. "vinyl", "shellac"). If provided,
                          _MATERIAL_GATE_DBFS[material_key] acts as the minimum gate.

    Returns:
        Gate threshold in dBFS. Frames above this threshold receive makeup gain.
    """
    _mat_floor = _MATERIAL_GATE_DBFS.get(str(material_key or "unknown").lower(), fallback_gate_dbfs)
    _floor = max(_mat_floor, fallback_gate_dbfs)
    try:
        arr = np.asarray(reference_audio, dtype=np.float32)
        if arr.ndim == 2:
            ch_first = arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]
            mono = np.mean(arr, axis=0) if ch_first else np.mean(arr, axis=1)
        else:
            mono = arr
        n = len(mono)
        n_full = max(1, n // frame_len)
        if n_full < 10:
            return _floor
        rms_db_vals: list[float] = []
        for fi in range(n_full):
            s, e = fi * frame_len, min((fi + 1) * frame_len, n)
            chunk = mono[s:e].astype(np.float64)
            rms_db_vals.append(float(20.0 * np.log10(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)) + 1e-12)))
        tail_s = n_full * frame_len
        if tail_s < n:
            tail = mono[tail_s:].astype(np.float64)
            rms_db_vals.append(float(20.0 * np.log10(float(np.sqrt(np.mean(tail * tail) + 1e-12)) + 1e-12)))
        if len(rms_db_vals) < 10:
            return _floor
        noise_floor_db = float(np.percentile(rms_db_vals, percentile))
        gate = float(np.clip(noise_floor_db + margin_db, -60.0, -10.0))
        return max(_floor, gate)  # material floor as minimum; signal can only raise it
    except Exception:
        return _floor


def _edge_channel_views(audio: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return [arr]
    try:
        left, right = stereo_channel_view(arr)
        return [
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
        ]
    except ValueError:
        return [safe_to_mono(arr)]


def _match_edge_channel_views(
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ref_channels = _edge_channel_views(reference_audio)
    cand_channels = _edge_channel_views(candidate_audio)
    n_channels = max(len(ref_channels), len(cand_channels))
    if len(ref_channels) == 1 and n_channels > 1:
        ref_channels = ref_channels * n_channels
    if len(cand_channels) == 1 and n_channels > 1:
        cand_channels = cand_channels * n_channels
    return ref_channels[:n_channels], cand_channels[:n_channels]


def _profile_channel_flags(
    profile: dict[str, float | int | bool] | None,
    key: str,
    fallback: bool,
    channel_count: int,
) -> list[bool]:
    if profile is None:
        return [fallback] * channel_count
    raw_flags = profile.get(key)
    if isinstance(raw_flags, (list, tuple)):
        flags = [bool(v) for v in raw_flags]
    else:
        flags = []
    if not flags:
        return [fallback] * channel_count
    if len(flags) < channel_count:
        flags.extend([flags[-1]] * (channel_count - len(flags)))
    return flags[:channel_count]


def _quiet_edge_guard_profile(
    reference_audio: np.ndarray,
    sr: int,
    *,
    material_key: str | None = None,
) -> dict[str, float | int | bool] | None:
    """Measure whether original intro/outro should be treated as quiet edges."""
    ref_arr = np.asarray(reference_audio, dtype=np.float32)
    ref = safe_to_mono(ref_arr)
    n = len(ref)
    if n < max(int(sr * 2.0), 4_800):
        return None

    edge_len = min(int(sr * 4.0), max(int(sr * 1.0), int(n * 0.10)))
    centre_len = min(int(sr * 4.0), max(int(sr * 1.0), int(n * 0.20)))
    centre_start = max(0, (n - centre_len) // 2)
    gate_dbfs = compute_signal_relative_gate_dbfs(
        ref,
        fallback_gate_dbfs=-36.0,
        material_key=material_key,
    )
    centre_ref_db = compute_gated_rms_dbfs(ref[centre_start : centre_start + centre_len], gate_dbfs=gate_dbfs)

    intro_ref_db = compute_gated_rms_dbfs(ref[:edge_len], gate_dbfs=gate_dbfs)
    outro_ref_db = compute_gated_rms_dbfs(ref[-edge_len:], gate_dbfs=gate_dbfs)
    intro_quiet = bool((intro_ref_db <= gate_dbfs + 3.0) or (intro_ref_db <= centre_ref_db - 6.0))
    outro_quiet = bool((outro_ref_db <= gate_dbfs + 3.0) or (outro_ref_db <= centre_ref_db - 6.0))

    intro_quiet_channels: list[bool] = []
    outro_quiet_channels: list[bool] = []
    for channel in _edge_channel_views(ref_arr):
        channel = channel[:n]
        centre_ch_db = compute_gated_rms_dbfs(channel[centre_start : centre_start + centre_len], gate_dbfs=gate_dbfs)
        intro_ch_db = compute_gated_rms_dbfs(channel[:edge_len], gate_dbfs=gate_dbfs)
        outro_ch_db = compute_gated_rms_dbfs(channel[-edge_len:], gate_dbfs=gate_dbfs)
        intro_quiet_channels.append(bool((intro_ch_db <= gate_dbfs + 3.0) or (intro_ch_db <= centre_ch_db - 6.0)))
        outro_quiet_channels.append(bool((outro_ch_db <= gate_dbfs + 3.0) or (outro_ch_db <= centre_ch_db - 6.0)))

    return {
        "n": n,
        "edge_len": edge_len,
        "gate_dbfs": gate_dbfs,
        "channel_count": len(intro_quiet_channels),
        "intro_quiet": bool(intro_quiet or any(intro_quiet_channels)),
        "outro_quiet": bool(outro_quiet or any(outro_quiet_channels)),
        "intro_quiet_channels": tuple(intro_quiet_channels),
        "outro_quiet_channels": tuple(outro_quiet_channels),
    }


def quiet_edge_boost_ok(
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sr: int,
    *,
    material_key: str | None = None,
    max_edge_boost_db: float = 2.0,
) -> bool:
    """Reject candidates that inflate intentionally quiet song edges."""
    profile = _quiet_edge_guard_profile(reference_audio, sr, material_key=material_key)
    if profile is None:
        return True

    ref_channels, cand_channels = _match_edge_channel_views(reference_audio, candidate_audio)
    n = min(
        int(profile["n"]),
        *(len(ch) for ch in ref_channels),
        *(len(ch) for ch in cand_channels),
    )
    if n < max(int(sr * 2.0), 4_800):
        return True

    edge_len = int(profile["edge_len"])
    gate_dbfs = float(profile["gate_dbfs"])
    intro_flags = _profile_channel_flags(
        profile,
        "intro_quiet_channels",
        bool(profile["intro_quiet"]),
        len(ref_channels),
    )
    outro_flags = _profile_channel_flags(
        profile,
        "outro_quiet_channels",
        bool(profile["outro_quiet"]),
        len(ref_channels),
    )

    def _p995_dbfs(x: np.ndarray) -> float:
        return float(20.0 * np.log10(float(np.percentile(np.abs(x.astype(np.float64)), 99.5)) + 1e-12))

    for start, end, channel_flags in (
        (0, edge_len, intro_flags),
        (n - edge_len, n, outro_flags),
    ):
        if not any(channel_flags):
            continue
        for channel_index, (ref_channel, cand_channel) in enumerate(zip(ref_channels, cand_channels)):
            if not channel_flags[channel_index]:
                continue
            ref_edge = ref_channel[:n][start:end]
            cand_edge = cand_channel[:n][start:end]
            ref_edge_db = compute_gated_rms_dbfs(ref_edge, gate_dbfs=gate_dbfs)
            cand_edge_db = compute_gated_rms_dbfs(cand_edge, gate_dbfs=gate_dbfs)
            if cand_edge_db > ref_edge_db + max_edge_boost_db:
                return False

            ref_edge_peak_db = _p995_dbfs(ref_edge)
            cand_edge_peak_db = _p995_dbfs(cand_edge)
            if cand_edge_peak_db > ref_edge_peak_db + max_edge_boost_db + 1.0:
                return False
    return True


def _scale_audio_region(
    audio: np.ndarray,
    start: int,
    end: int,
    scale: float,
    channel_index: int | None = None,
) -> np.ndarray:
    if scale >= 0.9999 or end <= start:
        return audio
    out = np.array(audio, dtype=np.float32, copy=True)
    if out.ndim == 1:
        out[start:end] *= np.float32(scale)
        return out
    ch_first = out.shape[0] <= 2 and out.shape[1] > out.shape[0]
    if ch_first:
        if channel_index is None:
            out[:, start:end] *= np.float32(scale)
        else:
            out[channel_index, start:end] *= np.float32(scale)
        return out
    if channel_index is None:
        out[start:end, :] *= np.float32(scale)
    else:
        out[start:end, channel_index] *= np.float32(scale)
    return out


def limit_quiet_edge_boost(
    reference_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sr: int,
    *,
    material_key: str | None = None,
    max_edge_boost_db: float = 2.0,
) -> np.ndarray:
    """Scale quiet intro/outro regions back toward the original edge level."""
    profile = _quiet_edge_guard_profile(reference_audio, sr, material_key=material_key)
    if profile is None:
        return np.asarray(candidate_audio, dtype=np.float32)

    out = np.asarray(candidate_audio, dtype=np.float32)
    ref_channels, cand_channels = _match_edge_channel_views(reference_audio, out)
    n = min(
        int(profile["n"]),
        *(len(ch) for ch in ref_channels),
        *(len(ch) for ch in cand_channels),
    )
    if n < max(int(sr * 2.0), 4_800):
        return out

    edge_len = int(profile["edge_len"])
    gate_dbfs = float(profile["gate_dbfs"])
    intro_flags = _profile_channel_flags(
        profile,
        "intro_quiet_channels",
        bool(profile["intro_quiet"]),
        len(ref_channels),
    )
    outro_flags = _profile_channel_flags(
        profile,
        "outro_quiet_channels",
        bool(profile["outro_quiet"]),
        len(ref_channels),
    )

    def _p995_dbfs(x: np.ndarray) -> float:
        return float(20.0 * np.log10(float(np.percentile(np.abs(x.astype(np.float64)), 99.5)) + 1e-12))

    for start, end, channel_flags in (
        (0, edge_len, intro_flags),
        (n - edge_len, n, outro_flags),
    ):
        if not any(channel_flags):
            continue
        for channel_index, (ref_channel, cand_channel) in enumerate(zip(ref_channels, cand_channels)):
            if not channel_flags[channel_index]:
                continue
            ref_edge = ref_channel[:n][start:end]
            cand_edge = cand_channel[:n][start:end]
            ref_edge_db = compute_gated_rms_dbfs(ref_edge, gate_dbfs=gate_dbfs)
            cand_edge_db = compute_gated_rms_dbfs(cand_edge, gate_dbfs=gate_dbfs)
            ref_edge_peak_db = _p995_dbfs(ref_edge)
            cand_edge_peak_db = _p995_dbfs(cand_edge)

            scale = 1.0
            if cand_edge_db > ref_edge_db + max_edge_boost_db:
                scale = min(scale, float(10.0 ** ((ref_edge_db + max_edge_boost_db - cand_edge_db) / 20.0)))
            if cand_edge_peak_db > ref_edge_peak_db + max_edge_boost_db + 1.0:
                scale = min(
                    scale,
                    float(10.0 ** ((ref_edge_peak_db + max_edge_boost_db + 1.0 - cand_edge_peak_db) / 20.0)),
                )
            out = _scale_audio_region(out, start, end, max(scale, 0.0), channel_index=channel_index)
            _, cand_channels = _match_edge_channel_views(reference_audio, out)
    return out


def apply_musical_gain_envelope(  # pylint: disable=too-many-positional-arguments
    audio: np.ndarray,
    gain: float,
    gate_dbfs: float = -36.0,
    crossfade_ms: float = 10.0,
    sr: int = 48000,
    reference_for_gate: np.ndarray | None = None,
    material_key: str | None = None,
) -> np.ndarray:
    """§2.45a-II: Apply makeup gain ONLY to musical frames, leaving silence at gain=1.0.

    Silence frames (frame RMS <= effective_gate_dbfs) remain at unity gain.
    A short crossfade (box-blur on the gate envelope) prevents hard clicks at
    music/silence boundaries.

    Adaptive gate (§2.45a noise-floor-aware, v9.12.2 — CEDAR/iZotope RX approach):
        effective_gate = max(gate_dbfs, compute_signal_relative_gate_dbfs(reference, material_key))

        ``compute_signal_relative_gate_dbfs`` uses P15+9 dB from the reference signal
        (CEDAR minimum-statistics: gate = measured_noise_floor + margin).  This replaces
        the old P5+10 heuristic which failed when songs have short quiet sections (P5 falls
        into the music region → P5+10 < -36 → gate stays at -36 → vinyl noise at -33 passes).

        When ``reference_for_gate`` is provided (pre-phase audio), the gate is estimated
        from the ORIGINAL signal's noise floor — critical for partially-denoised audio
        where the processed audio's P15 drops to -55+ dBFS (noise removed), making the
        gate too low and letting residual noise at -35 dBFS be amplified → Pegelexplosion.

        When ``reference_for_gate`` is None, ``audio`` itself is used as reference
        (self-adaptive: dirty audio → high gate; clean audio → gate ≈ gate_dbfs).

    Args:
        audio:              Input audio (1D or 2D float32).
        gain:               Linear gain factor (>= 1.0; values <= 1.0005 are skipped).
        gate_dbfs:          Floor threshold — effective gate can only be equal to or HIGHER.
                            Acts as the hard minimum (material floor guarantee).
        crossfade_ms:       Width of the smoothing window at music/silence transitions.
        sr:                 Sample rate used to convert crossfade_ms to samples.
        reference_for_gate: Optional pre-phase audio for noise-floor estimation.
                            When set, the gate is derived from this signal (not from audio).
        material_key:       Optional material type (e.g. "vinyl", "shellac").
                            Used as minimum floor in compute_signal_relative_gate_dbfs.

    Returns:
        Audio with gain applied only on musical frames, same shape and dtype.
    """
    # Scalar early-exit only — array gain passes through (broadcast in per_sample_gain)
    if np.ndim(gain) == 0 and float(gain) <= 1.0005:
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

    # --- Adaptive gate: compute signal-relative threshold (CEDAR/iZotope RX approach) ---
    # §2.45a v9.12.2: effective_gate = max(gate_dbfs, compute_signal_relative_gate_dbfs(...))
    # Uses P15+9 dB from the reference signal (more robust than P5+10 for loud songs with
    # short quiet sections).  Reference priority: reference_for_gate > audio itself.
    _gate_ref = reference_for_gate if reference_for_gate is not None else audio
    effective_gate = compute_signal_relative_gate_dbfs(
        _gate_ref,
        material_key=material_key,
        fallback_gate_dbfs=gate_dbfs,
    )
    # gate_dbfs is the hard floor — signal-relative gate can only raise it, never lower.
    effective_gate = max(gate_dbfs, effective_gate)

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

    def _render(gain_env: np.ndarray) -> np.ndarray:
        if was_2d:
            ch_first = arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]
            if ch_first:
                return cast(np.ndarray, np.asarray(arr * gain_env[np.newaxis, :], dtype=np.float32))
            return cast(np.ndarray, np.asarray(arr * gain_env[:, np.newaxis], dtype=np.float32))
        return cast(np.ndarray, np.asarray(arr * gain_env, dtype=np.float32))

    out = _render(per_sample_gain)
    edge_reference = reference_for_gate if reference_for_gate is not None else arr
    edge_profile = _quiet_edge_guard_profile(edge_reference, sr, material_key=material_key)
    if edge_profile is not None and not quiet_edge_boost_ok(
        edge_reference,
        out,
        sr,
        material_key=material_key,
    ):
        edge_len = int(edge_profile["edge_len"])
        if bool(edge_profile["intro_quiet"]):
            per_sample_gain[:edge_len] = np.minimum(per_sample_gain[:edge_len], 1.0)
        if bool(edge_profile["outro_quiet"]):
            per_sample_gain[-edge_len:] = np.minimum(per_sample_gain[-edge_len:], 1.0)
        out = _render(per_sample_gain)
        out = limit_quiet_edge_boost(
            edge_reference,
            out,
            sr,
            material_key=material_key,
        )
    return out


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
