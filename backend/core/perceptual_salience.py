"""Perceptual salience estimator for defect locations.

Assigns a perceptual salience score (0.0–1.0) to each defect event based on
psychoacoustic masking models.  Defects masked by louder surrounding content
score low; defects in quiet/exposed passages score high.

Scientific basis:
- Simultaneous masking: Fastl & Zwicker (2007) "Psychoacoustics: Facts and Models"
- Temporal masking: forward masking ~200 ms, backward masking ~20 ms (ISO 226:2023)
- Loudness model: ITU-R BS.1770-5 momentary loudness (400 ms windows)

The estimator does NOT modify audio — it only annotates DefectScore metadata
with a ``perceptual_salience`` field that downstream stages can use to:
1. Prioritize repair of perceptually salient defects
2. Skip repair of masked (inaudible) defects to reduce artefact risk
3. Report to the user which defects were audible

Module invariants (§3.x compliant):
- Thread-safe singleton via double-checked locking
- NaN/Inf guard on all numeric outputs
- No sample-rate assertion (analysis module — works at native import SR)
- English docstrings and log messages
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading

import numpy as np

from backend.core.defect_scanner import DefectAnalysisResult, DefectType

logger = logging.getLogger(__name__)


@dataclass
class SalienceAnnotation:
    """Salience annotation for a single defect event."""

    defect_type: DefectType
    location: tuple[float, float]  # (start_s, end_s)
    salience: float  # 0.0 = completely masked, 1.0 = fully exposed
    local_loudness_lufs: float  # momentary loudness at defect location
    surrounding_loudness_lufs: float  # loudness of masking context (±400 ms)
    masking_type: str  # "simultaneous" | "temporal_forward" | "temporal_backward" | "none"


@dataclass
class SalienceResult:
    """Result of perceptual salience analysis for all defects."""

    annotations: list[SalienceAnnotation] = field(default_factory=list)
    mean_salience: float = 0.0
    n_salient: int = 0  # events with salience >= 0.5
    n_masked: int = 0  # events with salience < 0.3


class PerceptualSalienceEstimator:
    """Estimates perceptual salience of detected defect events.

    Uses momentary loudness (ITU-R BS.1770-5, 400 ms windows) to determine
    whether a defect is perceptually masked by surrounding audio content.

    Masking model:
    - Simultaneous: defect during loud passage (defect loudness < context - 12 dB)
    - Temporal forward: defect within 200 ms after loud transient (context - 8 dB)
    - Temporal backward: defect within 20 ms before loud passage (context - 6 dB)
    """

    _WINDOW_S = 0.4  # ITU-R BS.1770-5 momentary loudness (400 ms)
    _HOP_S = 0.1  # 100 ms hop for loudness profile
    _FORWARD_MASK_S = 0.200  # forward masking duration (200 ms)
    _BACKWARD_MASK_S = 0.020  # backward masking duration (20 ms)
    _SIMULTANEOUS_THRESHOLD_DB = 12.0  # dB below context = masked
    _FORWARD_THRESHOLD_DB = 8.0
    _BACKWARD_THRESHOLD_DB = 6.0

    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        defect_result: DefectAnalysisResult,
    ) -> SalienceResult:
        """Annotate all defect events with perceptual salience scores.

        Parameters
        ----------
        audio : np.ndarray
            Mono or stereo audio at native sample rate.
        sr : int
            Sample rate in Hz.
        defect_result : DefectAnalysisResult
            Output of DefectScanner.scan() with locations.

        Returns
        -------
        SalienceResult with per-event annotations.
        """
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        mono = np.nan_to_num(mono.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # Build momentary loudness profile (ITU-R BS.1770-5 simplified: RMS in dBFS)
        loudness_profile = self._compute_loudness_profile(mono, sr)
        duration_s = len(mono) / sr

        annotations: list[SalienceAnnotation] = []

        for defect_type, defect_score in defect_result.scores.items():
            if not defect_score.locations:
                continue
            for loc_start, loc_end in defect_score.locations:
                salience, local_lufs, context_lufs, mask_type = self._score_event(
                    loudness_profile,
                    sr,
                    duration_s,
                    loc_start,
                    loc_end,
                )
                annotations.append(
                    SalienceAnnotation(
                        defect_type=defect_type,
                        location=(loc_start, loc_end),
                        salience=salience,
                        local_loudness_lufs=local_lufs,
                        surrounding_loudness_lufs=context_lufs,
                        masking_type=mask_type,
                    )
                )

        mean_sal = float(np.mean([a.salience for a in annotations])) if annotations else 0.0
        n_salient = sum(1 for a in annotations if a.salience >= 0.5)
        n_masked = sum(1 for a in annotations if a.salience < 0.3)

        result = SalienceResult(
            annotations=annotations,
            mean_salience=float(np.nan_to_num(mean_sal, nan=0.0)),
            n_salient=n_salient,
            n_masked=n_masked,
        )

        logger.info(
            "PerceptualSalience: %d events analysed, %d salient (>=0.5), %d masked (<0.3), mean=%.3f",
            len(annotations),
            n_salient,
            n_masked,
            result.mean_salience,
        )
        return result

    def annotate_defect_scores(
        self,
        audio: np.ndarray,
        sr: int,
        defect_result: DefectAnalysisResult,
    ) -> DefectAnalysisResult:
        """Annotate DefectAnalysisResult in-place with salience metadata.

        Adds to each DefectScore.metadata:
        - ``perceptual_salience``: mean salience across events (0.0–1.0)
        - ``n_salient_events``: count of events with salience >= 0.5
        - ``n_masked_events``: count of events with salience < 0.3

        Additionally scales severity by mean salience:
        ``adjusted_severity = severity * (0.3 + 0.7 * mean_salience)``
        This preserves a base severity (30%) even for fully masked defects
        while boosting exposed defects to near-original severity.
        """
        salience_result = self.estimate(audio, sr, defect_result)

        # Group annotations by defect type
        by_type: dict[DefectType, list[SalienceAnnotation]] = {}
        for ann in salience_result.annotations:
            by_type.setdefault(ann.defect_type, []).append(ann)

        for dt, annotations in by_type.items():
            if dt not in defect_result.scores:
                continue
            ds = defect_result.scores[dt]
            mean_sal = float(np.mean([a.salience for a in annotations]))
            mean_sal = float(np.nan_to_num(mean_sal, nan=0.5))
            ds.metadata["perceptual_salience"] = round(mean_sal, 3)
            ds.metadata["n_salient_events"] = sum(1 for a in annotations if a.salience >= 0.5)
            ds.metadata["n_masked_events"] = sum(1 for a in annotations if a.salience < 0.3)

            # Scale severity: masked defects get reduced priority
            old_sev = ds.severity
            ds.severity = float(
                np.nan_to_num(
                    min(1.0, old_sev * (0.3 + 0.7 * mean_sal)),
                    nan=0.0,
                )
            )
            if abs(ds.severity - old_sev) > 0.01:
                logger.debug(
                    "Salience adjustment: %s severity %.3f → %.3f (salience=%.3f)",
                    dt.value,
                    old_sev,
                    ds.severity,
                    mean_sal,
                )

        return defect_result

    # ------------------------------------------------------------------
    # Internal: Loudness profile
    # ------------------------------------------------------------------

    def _compute_loudness_profile(
        self,
        mono: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Compute momentary loudness profile (dBFS, 400 ms windows, 100 ms hop).

        Returns array of shape (n_frames,) with loudness in dBFS per frame.
        """
        win_samples = max(1, int(self._WINDOW_S * sr))
        hop_samples = max(1, int(self._HOP_S * sr))

        n_frames = max(1, (len(mono) - win_samples) // hop_samples + 1)
        loudness = np.full(n_frames, -100.0, dtype=np.float64)

        for i in range(n_frames):
            start = i * hop_samples
            end = start + win_samples
            if end > len(mono):
                break
            rms = np.sqrt(np.mean(mono[start:end] ** 2) + 1e-12)
            loudness[i] = 20.0 * np.log10(max(rms, 1e-10))

        return loudness

    def _time_to_frame(self, t: float, sr: int) -> int:
        """Convert time in seconds to loudness profile frame index."""
        hop_samples = max(1, int(self._HOP_S * sr))
        return max(0, int(t * sr / hop_samples))

    def _score_event(
        self,
        loudness_profile: np.ndarray,
        sr: int,
        duration_s: float,
        loc_start: float,
        loc_end: float,
    ) -> tuple[float, float, float, str]:
        """Score a single defect event for perceptual salience.

        Returns (salience, local_lufs, context_lufs, masking_type).
        """
        n_frames = len(loudness_profile)
        if n_frames == 0:
            return 1.0, -100.0, -100.0, "none"

        # Frame indices for the defect location
        f_start = min(self._time_to_frame(loc_start, sr), n_frames - 1)
        f_end = min(self._time_to_frame(loc_end, sr), n_frames - 1)
        f_end = max(f_end, f_start)

        # Local loudness at defect location
        local_lufs = float(np.max(loudness_profile[f_start : f_end + 1]))

        # Context: surrounding ±400 ms window (excluding the defect itself)
        ctx_start_t = max(0.0, loc_start - self._WINDOW_S)
        ctx_end_t = min(duration_s, loc_end + self._WINDOW_S)
        cf_start = min(self._time_to_frame(ctx_start_t, sr), n_frames - 1)
        cf_end = min(self._time_to_frame(ctx_end_t, sr), n_frames - 1)

        # Context frames excluding defect region
        ctx_frames = np.concatenate(
            [
                loudness_profile[cf_start:f_start],
                loudness_profile[f_end + 1 : cf_end + 1],
            ]
        )
        if len(ctx_frames) == 0:
            return 1.0, local_lufs, local_lufs, "none"

        context_lufs = float(np.max(ctx_frames))

        # Check masking conditions
        diff_db = context_lufs - local_lufs

        # Forward masking: loud content just before the defect
        fwd_start_t = max(0.0, loc_start - self._FORWARD_MASK_S)
        ff_start = min(self._time_to_frame(fwd_start_t, sr), n_frames - 1)
        pre_lufs = float(np.max(loudness_profile[ff_start : f_start + 1])) if ff_start < f_start else -100.0

        # Backward masking: loud content just after the defect
        bwd_end_t = min(duration_s, loc_end + self._BACKWARD_MASK_S)
        bf_end = min(self._time_to_frame(bwd_end_t, sr), n_frames - 1)
        post_lufs = float(np.max(loudness_profile[f_end : bf_end + 1])) if f_end < bf_end else -100.0

        masking_type = "none"
        salience = 1.0

        # Simultaneous masking (context louder than defect by threshold)
        if diff_db >= self._SIMULTANEOUS_THRESHOLD_DB:
            masking_type = "simultaneous"
            # Salience decreases with increasing masking margin
            salience = max(0.0, 1.0 - (diff_db - self._SIMULTANEOUS_THRESHOLD_DB) / 20.0)

        # Forward masking (loud content before defect)
        elif (pre_lufs - local_lufs) >= self._FORWARD_THRESHOLD_DB:
            masking_type = "temporal_forward"
            margin = pre_lufs - local_lufs - self._FORWARD_THRESHOLD_DB
            salience = max(0.0, 1.0 - margin / 15.0)

        # Backward masking (loud content after defect)
        elif (post_lufs - local_lufs) >= self._BACKWARD_THRESHOLD_DB:
            masking_type = "temporal_backward"
            margin = post_lufs - local_lufs - self._BACKWARD_THRESHOLD_DB
            salience = max(0.0, 1.0 - margin / 15.0)

        salience = float(np.nan_to_num(np.clip(salience, 0.0, 1.0), nan=0.5))
        return salience, local_lufs, context_lufs, masking_type


# ---------------------------------------------------------------------------
# Thread-safe singleton (double-checked locking — §3.2)
# ---------------------------------------------------------------------------

_instance: PerceptualSalienceEstimator | None = None
_lock = threading.Lock()


def get_perceptual_salience_estimator() -> PerceptualSalienceEstimator:
    """Return thread-safe singleton PerceptualSalienceEstimator."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PerceptualSalienceEstimator()
    return _instance
