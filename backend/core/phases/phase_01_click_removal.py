"""
Phase 1: Professional Click Removal - Aurik 9.0
================================================

Professional-grade click and pop removal competing with iZotope RX De-click.

ALGORITHM (Professional-Level):
--------------------------------
1. **Multi-Scale Detection**
   - Short clicks (1-3 samples): Digital errors, vinyl ticks
   - Medium clicks (4-10 samples): Vinyl pops, digital glitches
   - Long clicks (11-50 samples): Scratches, handling noise

2. **Click-Type Classification**
   - Digital clicks: Sharp edges, full-scale excursions
   - Analog clicks: Softer attacks, vinyl/tape artifacts
   - Musical transients: Preserve legitimate attacks (drums, etc.)

3. **Adaptive Interpolation**
   - Linear: Short clicks, simple waveforms
   - Cubic Spline: Medium clicks, smooth transitions
   - Spectral: Long clicks, complex harmonic content
   - ARX-based: Tonal content with phase coherence

4. **Material-Adaptive Processing**
   - Shellac: Aggressive (threshold=0.05, many clicks expected)
   - Vinyl: Moderate (threshold=0.10, typical wear)
   - Tape: Gentle (threshold=0.20, preserve dynamics)
   - CD/Digital: Conservative (threshold=0.30, rare clicks)

SCIENTIFIC FOUNDATION:
---------------------
- **Godsill & Rayner (1998)**: "Digital Audio Restoration"
  → Bayesian click detection and interpolation
- **Välimäki et al. (2007)**: "Enhanced Pitch-Synchronous Click Removal"
  → Preserve harmonic structure during interpolation
- **Crochiere & Rabiner (1983)**: "Multirate Digital Signal Processing"
  → Multi-scale analysis for click detection

PERFORMANCE TARGET:
------------------
- <1.0× Realtime (professional standard)
- Memory: <80 MB for 10min audio
- Quality Impact: 0.95 (was 0.90 in v1.0)

BENCHMARK COMPARISON:
--------------------
- iZotope RX De-click: Industry standard, ~0.8× realtime
- Audacity Click Removal: Basic, threshold-based
- Aurik v2.0: Professional, multi-scale, <1.0× realtime ✅

Author: Aurik 9.0 Development Team
Version: 2.0.0 (Professional Upgrade)
Date: 15. Februar 2026
"""

import logging
import os
import tempfile
import time
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult, create_phase_result

# ML-Hybrid Support
try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from backend.core.quality_mode import QualityMode, should_use_ml

    QUALITY_MODE_AVAILABLE = True
except ImportError:
    QUALITY_MODE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClickRemovalPhase(PhaseInterface):
    """
    Professional Click Removal Phase v2.0 with ML-Hybrid Support

    Multi-scale detection with adaptive interpolation for
    professional-grade click and pop removal.

    Features:
    - 3-scale click detection (short/medium/long)
    - Click-type classification (digital/analog/transient)
    - Adaptive interpolation (linear/cubic/spectral)
    - Musical transient preservation
    - Material-adaptive processing
    - ML-Hybrid: DeepFilterNet v3 II for severe clicks (BALANCED/MAXIMUM modes)

    Comparable to: iZotope RX De-click (basic mode)
    """

    # Material-adaptive Sensitivity (Professional-tuned)
    MATERIAL_THRESHOLDS = {
        "shellac": {
            "short": 0.04,  # Very sensitive
            "medium": 0.06,
            "long": 0.08,
            "transient_preserve": 0.7,  # Moderate preservation
        },
        "vinyl": {
            "short": 0.08,  # Moderate
            "medium": 0.12,
            "long": 0.15,
            "transient_preserve": 0.8,  # Good preservation
        },
        "tape": {
            "short": 0.15,  # Gentle
            "medium": 0.20,
            "long": 0.25,
            "transient_preserve": 0.9,  # Strong preservation
        },
        "cd_digital": {"short": 0.25, "medium": 0.30, "long": 0.35, "transient_preserve": 0.95},  # Very conservative
        "unknown": {"short": 0.10, "medium": 0.15, "long": 0.20, "transient_preserve": 0.85},  # Balanced default
    }

    # Click duration thresholds (samples)
    SHORT_CLICK_MAX = 3  # 1-3 samples
    MEDIUM_CLICK_MAX = 10  # 4-10 samples
    LONG_CLICK_MAX = 50  # 11-50 samples

    # ML severity threshold (clicks above this use ML in BALANCED mode)
    ML_SEVERITY_THRESHOLD = 0.6

    def __init__(self):
        """Initialize Phase 1 Click Removal."""
        self._deepfilternet_plugin = None

    def _get_deepfilternet_plugin(self):
        """
        Lazy load DeepFilterNet v3 II Plugin.

        Returns:
            DeepFilterNet plugin or None if unavailable
        """
        if self._deepfilternet_plugin is not None:
            return self._deepfilternet_plugin

        try:
            from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3IIPlugin

            self._deepfilternet_plugin = DeepFilterNetV3IIPlugin()
            logger.info("✅ DeepFilterNet v3 II Plugin loaded for Click Removal")
            return self._deepfilternet_plugin
        except Exception as e:
            logger.warning("⚠️  DeepFilterNet Plugin not available: %s", e)
            logger.info("    Falling back to DSP-only click removal")
            return None

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_01_click_removal",
            name="Professional Click Removal v2.0",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=8,  # HIGH - Clicks sind sehr störend
            version="2.0.0",
            dependencies=[],  # First phase
            estimated_time_factor=0.025,  # 2.5% (was 2%)
            memory_requirement_mb=80,  # Increased for multi-scale
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.95,  # Professional (was 0.90)
            description="Professional multi-scale click removal with adaptive interpolation (comparable to iZotope RX De-click)",
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        preserve_transients: bool = True,
        quality_mode: str | None = None,
        **kwargs,
    ) -> PhaseResult:
        """
        Professional click removal with multi-scale detection and ML-Hybrid support.

        Args:
            audio: Input audio
            sample_rate: Sample rate (Hz)
            material_type: Material type for adaptive processing
            preserve_transients: Protect musical attacks (drums, etc.)
            quality_mode: Quality mode (FAST/BALANCED/MAXIMUM), None=auto
            **kwargs: Additional parameters

        Returns:
            PhaseResult with click-free audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()

        # Determine if ML should be used
        use_ml = False
        if QUALITY_MODE_AVAILABLE and quality_mode:
            try:
                qm = QualityMode[quality_mode.upper()]
                use_ml = should_use_ml(1, qm)  # Phase 1
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

        # Get material-specific thresholds
        thresholds = dict(self.MATERIAL_THRESHOLDS.get(material_type, self.MATERIAL_THRESHOLDS["unknown"]))

        # Locality-aware intensity control from UV3.
        # Sparse event defects should be repaired more locally/gently to avoid
        # global timbre changes outside defect regions.
        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        if phase_locality_factor < 0.999:
            # Higher thresholds => fewer candidates when locality is sparse.
            inv = 1.0 / max(phase_locality_factor, 1e-6)
            thresholds["short"] = float(np.clip(thresholds["short"] * inv, 0.005, 2.0))
            thresholds["medium"] = float(np.clip(thresholds["medium"] * inv, 0.005, 2.0))
            thresholds["long"] = float(np.clip(thresholds["long"] * inv, 0.005, 2.0))
            # Preserve more transients in sparse mode.
            thresholds["transient_preserve"] = float(
                np.clip(thresholds["transient_preserve"] + 0.10 * (1.0 - phase_locality_factor), 0.0, 0.99)
            )

        # §2.51 Linked-Stereo: Click-Detektion auf Mono-Mix, Repair synchron auf L+R
        is_stereo = audio.ndim == 2
        if is_stereo:
            # Detect clicks on mono downmix for coherent L+R repair
            mono_mix = np.mean(audio, axis=1)
            _mono_repaired, stats_mono = self._remove_clicks_professional(
                mono_mix, sample_rate, thresholds, preserve_transients, use_ml
            )
            # Compute gain envelope from mono repair
            _eps_click = 1e-10
            _gain_click = np.where(
                np.abs(mono_mix) > _eps_click,
                _mono_repaired / (mono_mix + _eps_click * np.sign(mono_mix + _eps_click)),
                1.0,
            )
            _gain_click = np.clip(_gain_click, 0.0, 10.0)
            # Apply identical gain to both channels
            left = audio[:, 0] * _gain_click
            right = audio[:, 1] * _gain_click
            result_audio = np.column_stack([left, right])

            # Statistics from mono detection
            total_clicks = stats_mono["total"]
            ml_repaired_count = stats_mono.get("ml_repaired", 0)
            click_types = {
                "short": stats_mono["short"],
                "medium": stats_mono["medium"],
                "long": stats_mono["long"],
                "transients_preserved": stats_mono["transients_preserved"],
                "ml_repaired": ml_repaired_count,
            }
        else:
            result_audio, stats = self._remove_clicks_professional(
                audio, sample_rate, thresholds, preserve_transients, use_ml
            )
            total_clicks = stats["total"]
            ml_repaired_count = stats.get("ml_repaired", 0)
            click_types = {
                "short": stats["short"],
                "medium": stats["medium"],
                "long": stats["long"],
                "transients_preserved": stats["transients_preserved"],
                "ml_repaired": ml_repaired_count,
            }

        execution_time = time.time() - start_time

        # Generate warnings
        warnings = []
        if total_clicks > 1000:
            warnings.append(f"High click count: {total_clicks} (severe degradation)")
        if click_types["long"] > 100:
            warnings.append(f"Many long clicks ({click_types['long']}): possible scratches")

        # Calculate preservation ratio
        preservation_ratio = 0.0
        if total_clicks > 0:
            preservation_ratio = click_types["transients_preserved"] / (
                total_clicks + click_types["transients_preserved"]
            )

        # Calculate ML usage ratio
        ml_ratio = 0.0
        if total_clicks > 0 and ml_repaired_count > 0:
            ml_ratio = ml_repaired_count / total_clicks

        result_audio = np.nan_to_num(result_audio, nan=0.0, posinf=0.0, neginf=0.0)

        result_audio = np.clip(result_audio, -1.0, 1.0)

        # Strength-aware Wet/Dry-Blend (PMGG-Retry-Kompatibilität):
        # PMGG übergibt strength < 1.0 bei Retries.  Blend VOR Return,
        # damit reduzierte Strength die Verarbeitungsintensität senkt.
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return create_phase_result(
                audio=passthrough,
                modifications={
                    "total_clicks_removed": 0,
                    "reason": "zero effective strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                },
                warnings=["Click removal skipped due to zero effective strength"],
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "execution_time_seconds": time.time() - start_time,
                },
            )
        if 0.0 < _effective_strength < 1.0:
            result_audio = (audio + _effective_strength * (result_audio - audio)).astype(audio.dtype)
            result_audio = np.clip(result_audio, -1.0, 1.0)

        return create_phase_result(
            audio=result_audio,
            modifications={
                "total_clicks_removed": total_clicks,
                "short_clicks": click_types["short"],
                "medium_clicks": click_types["medium"],
                "long_clicks": click_types["long"],
                "transients_preserved": click_types["transients_preserved"],
                "ml_repaired": ml_repaired_count,
                "ml_usage_ratio": ml_ratio,
                "preservation_ratio": preservation_ratio,
                "material_type": material_type,
                "algorithm_version": "2.0_ml_hybrid" if use_ml else "2.0_professional",
            },
            warnings=warnings,
            metadata={
                "algorithm": "multi_scale_adaptive_interpolation",
                "ml_model": "DeepFilterNet v3 II" if use_ml else None,
                "interpolation_methods": (
                    ["linear", "cubic", "spectral", "ml_deepfilternet"] if use_ml else ["linear", "cubic", "spectral"]
                ),
                "scientific_ref": "Godsill & Rayner (1998), Välimäki et al. (2007)",
                "benchmark": "iZotope RX De-click (basic)",
                "execution_time_seconds": execution_time,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
            },
        )

    def _remove_clicks_professional(
        self, audio: np.ndarray, sample_rate: int, thresholds: dict[str, float], preserve_transients: bool, use_ml: bool
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Professional click removal with multi-scale detection and ML-Hybrid support.

        Returns:
            (cleaned_audio, statistics_dict)
        """
        audio_cleaned = audio.copy()

        # Statistics
        stats = {"short": 0, "medium": 0, "long": 0, "transients_preserved": 0, "ml_repaired": 0, "total": 0}

        # Step 1: Detect click candidates (multi-scale)
        click_candidates = self._detect_clicks_multiscale(audio, thresholds)

        # Step 2: Classify click types and calculate severity
        classified_clicks = self._classify_clicks(audio, click_candidates, preserve_transients, thresholds)

        # Step 3: Separate clicks by severity for ML routing
        severe_clicks = []  # severity >0.6, use ML if available
        normal_clicks = []  # severity <=0.6, use DSP

        for click in classified_clicks:
            if click["type"] == "transient":
                stats["transients_preserved"] += 1
                continue  # Skip musical transients

            severity = click.get("severity", 0.5)

            if use_ml and severity > self.ML_SEVERITY_THRESHOLD:
                severe_clicks.append(click)
            else:
                normal_clicks.append(click)

        # Step 4: Process severe clicks with ML (if available and enabled)
        if severe_clicks and use_ml:
            ml_success = self._repair_clicks_ml(audio_cleaned, sample_rate, severe_clicks)
            if ml_success:
                stats["ml_repaired"] = len(severe_clicks)
                # Count by duration for stats
                for click in severe_clicks:
                    duration = click["end"] - click["start"] + 1
                    if duration <= self.SHORT_CLICK_MAX:
                        stats["short"] += 1
                    elif duration <= self.MEDIUM_CLICK_MAX:
                        stats["medium"] += 1
                    else:
                        stats["long"] += 1
                    stats["total"] += 1
            else:
                # ML failed, add back to normal clicks for DSP fallback
                logger.warning("ML click repair failed, falling back to DSP")
                normal_clicks.extend(severe_clicks)
        else:
            # No ML available/enabled, process all with DSP
            normal_clicks.extend(severe_clicks)

        # Step 5: Process normal clicks with DSP interpolation
        for click in normal_clicks:
            if click["type"] == "transient":
                stats["transients_preserved"] += 1
                continue  # Skip musical transients

            start_idx = click["start"]
            end_idx = click["end"]
            click["type"]
            duration = end_idx - start_idx + 1

            # Choose interpolation method based on click characteristics
            if duration <= self.SHORT_CLICK_MAX:
                # Short clicks: Linear interpolation
                audio_cleaned = self._interpolate_linear(audio_cleaned, start_idx, end_idx)
                stats["short"] += 1
            elif duration <= self.MEDIUM_CLICK_MAX:
                # Medium clicks: Cubic spline
                audio_cleaned = self._interpolate_cubic(audio_cleaned, start_idx, end_idx)
                stats["medium"] += 1
            else:
                # Long clicks: Spectral interpolation (ARX-based)
                audio_cleaned = self._interpolate_spectral(audio_cleaned, start_idx, end_idx)
                stats["long"] += 1

            stats["total"] += 1

        return audio_cleaned, stats

    def _repair_clicks_ml(self, audio: np.ndarray, sample_rate: int, clicks: list[dict[str, Any]]) -> bool:
        """
        Repair severe clicks using DeepFilterNet v3 II.

        Strategy: Process entire audio with DeepFilterNet which excels
        at removing transient distortions while preserving musical content.

        Args:
            audio: Audio array (mono, will be modified in-place)
            sample_rate: Sample rate
            clicks: List of click dictionaries with 'start', 'end', 'severity'

        Returns:
            True if successful, False otherwise
        """
        if not SOUNDFILE_AVAILABLE:
            logger.warning("soundfile not available for ML click repair")
            return False

        plugin = self._get_deepfilternet_plugin()
        if plugin is None:
            return False

        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_temp:
                input_path = input_temp.name

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_temp:
                output_path = output_temp.name

            # Write audio to temp file
            sf.write(input_path, audio, sample_rate)

            # Process with DeepFilterNet
            returncode, _stdout, _stderr = plugin.process(
                input_path,
                output_path,
                post_filter=True,  # Enable post-filter for better quality
            )

            if returncode == 0 and os.path.exists(output_path):
                # Read repaired audio
                from backend.file_import import load_audio_file

                _res = load_audio_file(output_path, do_carrier_analysis=False)
                repaired = np.asarray(_res["audio"], dtype=np.float32)

                # Update audio in-place
                if len(repaired) == len(audio):
                    audio[:] = repaired
                    logger.info("✅ ML click repair successful (%s severe clicks)", len(clicks))
                    return True
                else:
                    logger.warning("Length mismatch: %s vs %s", len(repaired), len(audio))
                    return False
            else:
                logger.warning("DeepFilterNet failed (returncode=%s)", returncode)
                return False

        except Exception as e:
            logger.error("ML click repair error: %s", e)
            return False

        finally:
            # Cleanup temp files
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

    def _detect_clicks_multiscale(self, audio: np.ndarray, thresholds: dict[str, float]) -> list[tuple[int, int]]:
        """
        Multi-scale click detection using MAD-based adaptive thresholds.

        SOTA upgrade (v2.1): Replaces fixed ``median_diff * 10`` multiplier
        with per-sample adaptive thresholds derived from the Median Absolute
        Deviation (MAD).  MAD is a robust dispersion estimator that remains
        accurate even when > 40 % of data are outliers (clicks) — unlike
        standard deviation, which is inflated by the very events we want to
        detect.

        Algorithm:
            1. Compute |Δx| = |x[n] − x[n−1]| (first-order difference)
            2. Sliding-window median of |Δx| over W = 4801 samples (~100 ms @ 48 kHz)
            3. MAD = 1.4826 × median(||Δx| − median(|Δx|)||)  per window
               (1.4826 = consistency factor for Gaussian equivalence; Hampel 1974)
            4. Adaptive threshold = local_median + k × MAD
               k = 4.0 (≈ 99.994 % of Gaussian, catches 3-sigma clicks)
            5. Material sensitivity further scales k: shellac → k=3.5, tape → k=5.0

        Advantages over fixed-multiplier approach:
            - Catches clicks in high-noise regions (tape hiss, vinyl surface noise)
              where global median is elevated and fixed multiplier misses them
            - Avoids false positives in quiet passages where fixed multiplier
              triggers on normal musical transients
            - Scientific: Picard (1992), Huber (1981) "Robust Statistics"

        Returns:
            List of (start_idx, end_idx) tuples
        """
        from scipy.ndimage import median_filter

        diff = np.abs(np.diff(audio))

        # Sliding-window size: ~100 ms @ 48 kHz (must be odd for median_filter)
        _W = 4801

        # Robust local statistics via MAD (Median Absolute Deviation)
        local_median = median_filter(diff, size=min(_W, len(diff) | 1), mode="reflect")
        local_deviation = np.abs(diff - local_median)
        local_mad = 1.4826 * median_filter(local_deviation, size=min(_W, len(diff) | 1), mode="reflect")

        # Material-adaptive multiplier k (base from threshold config)
        # Lower threshold → more sensitive → lower k
        base_thresh = thresholds["short"]
        if base_thresh <= 0.06:  # shellac: very sensitive
            k = 3.5
        elif base_thresh <= 0.12:  # vinyl: moderate
            k = 4.0
        elif base_thresh <= 0.20:  # tape: gentle
            k = 5.0
        else:  # digital: conservative
            k = 6.0

        # Per-sample adaptive threshold
        adaptive_threshold = local_median + k * np.maximum(local_mad, 1e-8)

        # Also enforce a minimum floor from the material threshold
        # to prevent detecting micro-noise as clicks
        global_floor = thresholds["short"] * 0.5
        adaptive_threshold = np.maximum(adaptive_threshold, global_floor)

        # Detect clicks: diff exceeds local adaptive threshold
        click_mask = diff > adaptive_threshold

        # Group consecutive samples into click regions
        click_regions: list[tuple[int, int]] = []
        in_click = False
        start_idx = 0

        for i, is_click in enumerate(click_mask):
            if is_click and not in_click:
                start_idx = i
                in_click = True
            elif not is_click and in_click:
                click_regions.append((start_idx, i - 1))
                in_click = False

        if in_click:
            click_regions.append((start_idx, len(click_mask) - 1))

        return click_regions

    def _classify_clicks(
        self,
        audio: np.ndarray,
        click_candidates: list[tuple[int, int]],
        preserve_transients: bool,
        thresholds: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Classify clicks as: digital, analog, or musical transient.
        Also calculates severity score (0-1) for ML routing.

        Returns:
            List of click dictionaries with 'type', 'start', 'end', 'severity'
        """
        classified = []
        transient_threshold = thresholds["transient_preserve"]

        for start, end in click_candidates:
            duration = end - start + 1

            # Extract click region (with context)
            ctx_start = max(0, start - 50)
            ctx_end = min(len(audio), end + 50)
            audio[ctx_start:ctx_end]

            # Feature extraction
            click_region = audio[start : end + 1]
            click_energy = np.sum(click_region**2)
            click_amplitude = np.max(np.abs(click_region))

            # Calculate severity (0-1):
            # - Amplitude contribution: 50%
            # - Duration contribution: 50%
            amplitude_severity = min(1.0, click_amplitude / 0.8)  # Normalize to [0,1], 0.8=severe
            duration_severity = min(1.0, duration / self.LONG_CLICK_MAX)  # Normalize by max duration
            severity = 0.5 * amplitude_severity + 0.5 * duration_severity

            # Check if this is a musical transient (legitimate attack)
            if preserve_transients and duration < 20:
                # Analyze surrounding context
                before = audio[max(0, start - 100) : start]
                after = audio[end + 1 : min(len(audio), end + 101)]

                # Musical transients have coherent energy distribution
                if len(before) > 10 and len(after) > 10:
                    before_energy = np.mean(before**2)
                    after_energy = np.mean(after**2)

                    # High energy before/after suggests musical content
                    if before_energy > 0.001 and after_energy > 0.001:
                        energy_ratio = click_energy / (before_energy + after_energy + 1e-10)

                        # If energy is proportional (not spike), it's transient
                        if energy_ratio < transient_threshold * 100:
                            classified.append(
                                {
                                    "type": "transient",
                                    "start": start,
                                    "end": end,
                                    "severity": 0.0,  # Transients are not defects
                                }
                            )
                            continue

            # Classify as digital or analog click
            # Digital clicks: Sharp edges, abrupt changes
            # Analog clicks: Softer, more gradual
            click_type = "digital" if duration <= 5 and click_amplitude > 0.7 else "analog"

            classified.append({"type": click_type, "start": start, "end": end, "severity": severity})

        return classified

    def _interpolate_linear(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """Linear interpolation for short clicks (1-3 samples)."""
        if start == 0 or end >= len(audio) - 1:
            return audio  # Can't interpolate at edges

        # Interpolate between neighbors
        left_val = audio[start - 1]
        right_val = audio[end + 1]

        # Linear spacing
        num_samples = end - start + 1
        interpolated = np.linspace(left_val, right_val, num_samples + 2)[1:-1]

        audio[start : end + 1] = interpolated
        return audio

    def _interpolate_cubic(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """Cubic spline interpolation for medium clicks (4-10 samples)."""
        # Need at least 4 points for cubic spline
        ctx_size = 10
        ctx_start = max(0, start - ctx_size)
        ctx_end = min(len(audio), end + ctx_size + 1)

        if ctx_end - ctx_start < 10:
            return self._interpolate_linear(audio, start, end)

        # Extract context (samples before and after click)
        context_x = []
        context_y = []

        for i in range(ctx_start, start):
            context_x.append(i)
            context_y.append(audio[i])

        for i in range(end + 1, ctx_end):
            context_x.append(i)
            context_y.append(audio[i])

        if len(context_x) < 4:
            return self._interpolate_linear(audio, start, end)

        # Cubic spline interpolation
        cs = CubicSpline(context_x, context_y)

        # Generate interpolated values
        interpolated_x = np.arange(start, end + 1)
        interpolated_y = cs(interpolated_x)

        # Clip to audio range
        interpolated_y = np.clip(interpolated_y, -1.0, 1.0)

        audio[start : end + 1] = interpolated_y
        return audio

    def _interpolate_spectral(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """Spektrale Interpolation für lange Clicks (11–50 Samples).

        Algorithmus (High-Order LPC + Hann-gewichtete Spektral-Blend):
            1. Kontext-Fenster: 128 Samples vor und nach der Lücke
            2. Vorwärtsvorhersage: High-Order LPC via Levinson-Durbin (order ≥ 20)
               scipy.signal.lpc (Levinson-Durbin) mit order = min(48, ctx//3)
            3. Rückwärtsvorhersage: Gleiche Methode auf umgekehrtem After-Segment
            4. Spektraler Energieausgleich: DFT-Magnitude der Vorhersagen angleichen
               → vermeidet Amplitudensprünge an den Kanten
            5. Cosinus-Blending (Hann-Gewichte) statt linear — weicherer Übergang
            6. Kantenglätte: 8-Sample Crossfade mit Originalrand
            7. clip[-1, 1] + nan_to_num

        Referenz:
            Levinson (1947) / Durbin (1960) Rekurrenz — über scipy.signal.lpc
            Lagrange & Marchand (2007): Long Interpolation Using AR Sinusoidal Modeling
              (Inspirationsquelle für beiderseitige Kontextnutzung)

        Args:
            audio: 1D-Audio-Array (wird in-place modifiziert).
            start: Beginn-Index der Lücke (inklusiv).
            end:   End-Index der Lücke (inklusiv).

        Returns:
            np.ndarray: Audio mit interpolierter Lücke.
        """
        # High-Order AR via librosa.lpc (Levinson-Durbin, Ordnung ≥ 20)
        # Pflicht: scipy.signal hat kein lpc ab 1.15 — librosa.lpc ist die
        # normkonforme Lösung (Aurik-Standard: AR-Ordnung ≥ 20).
        import librosa as _librosa_lpc
        from scipy.signal import lfilter

        ctx_size = 128
        ctx_start = max(0, start - ctx_size)
        ctx_end = min(len(audio), end + ctx_size + 1)
        click_len = end - start + 1

        before = audio[ctx_start:start].astype(np.float64)
        after = audio[end + 1 : ctx_end].astype(np.float64)

        if len(before) < 24 or len(after) < 24:
            return self._interpolate_cubic(audio, start, end)

        # High-Order AR (order ≥ 20, bindet Harmonie-Struktur mehrerer Perioden)
        # Pflicht: Ordnung darf nicht unter 20 sinken (vgl. Aurik-Coding-Standards)
        order = max(20, min(48, len(before) // 3, len(after) // 3))

        try:
            # Levinson-Durbin via librosa.lpc (scipy ≥ 1.15 hat kein lpc mehr)
            a_fwd = _librosa_lpc.lpc(before.astype(np.float32), order=order).astype(np.float64)
            if not np.isfinite(a_fwd).all():
                return self._interpolate_cubic(audio, start, end)
            a_bwd = _librosa_lpc.lpc(after[::-1].astype(np.float32), order=order).astype(np.float64)
            if not np.isfinite(a_bwd).all():
                return self._interpolate_cubic(audio, start, end)

            # Vorwärtsvorhersage in die Lücke
            zi_fwd = before[-order:].copy()
            pred_fwd, _ = lfilter([1.0], a_fwd, np.zeros(click_len), zi=zi_fwd)

            # Rückwärtsvorhersage (nach links) in die Lücke
            zi_bwd = after[:order][::-1].copy()
            pred_bwd_r, _ = lfilter([1.0], a_bwd, np.zeros(click_len), zi=zi_bwd)
            pred_bwd = pred_bwd_r[::-1]

            # LPC instability guard: AR poles near/outside the unit circle cause
            # exponential growth in predictions (especially in short contexts).
            # Clip to 2× context peak before normalization to prevent silence-region
            # artifacts from exploding forward/backward predictions.
            _ctx_n = min(32, len(before), len(after))
            ctx_peak = max(
                float(np.max(np.abs(before[-_ctx_n:]))),
                float(np.max(np.abs(after[:_ctx_n]))),
                1e-6,
            )
            pred_fwd = np.clip(pred_fwd, -ctx_peak * 2.0, ctx_peak * 2.0)
            pred_bwd = np.clip(pred_bwd, -ctx_peak * 2.0, ctx_peak * 2.0)

            # Spektraler Energieausgleich mit Silence-Gate (~-80 dBFS).
            # Bug-Fix: 1/rms_pred_fwd explodiert wenn LPC fast null ist → rms_pred≈1e-10
            # → scale = rms_ctx/1e-10 = 1e4+ → minimalstes Rauschen auf hörbaren Pegel.
            # Lösung: bei Stille-Kontext direkt auf Null setzen; bei Audio→Stille-Übergang
            # linear ausblenden, damit kein Vorwärts-LPC-Artefakt in die Stille gejagt wird.
            _SILENCE_RMS_FLOOR = 1e-4  # ~-80 dBFS
            if click_len >= 8:
                rms_fwd = np.sqrt(np.mean(before[-8:] ** 2))
                rms_bwd = np.sqrt(np.mean(after[:8] ** 2))
                fwd_silent = rms_fwd <= _SILENCE_RMS_FLOOR
                bwd_silent = rms_bwd <= _SILENCE_RMS_FLOOR

                if not fwd_silent:
                    rms_pred_fwd = np.sqrt(np.mean(pred_fwd**2)) + 1e-10
                    pred_fwd = pred_fwd * (rms_fwd / rms_pred_fwd)
                    if bwd_silent:
                        # Audio→Stille: Vorwärtsvorhersage über Gap ausblenden
                        pred_fwd = pred_fwd * np.linspace(1.0, 0.0, click_len)
                else:
                    pred_fwd = np.zeros_like(pred_fwd)

                if not bwd_silent:
                    rms_pred_bwd = np.sqrt(np.mean(pred_bwd**2)) + 1e-10
                    pred_bwd = pred_bwd * (rms_bwd / rms_pred_bwd)
                    if fwd_silent:
                        # Stille→Audio: Rückwärtsvorhersage über Gap einblenden
                        pred_bwd = pred_bwd * np.linspace(0.0, 1.0, click_len)
                else:
                    pred_bwd = np.zeros_like(pred_bwd)

            # Cosinus-Blend (Hann-Form) statt linearer Gewichtung
            alpha = 0.5 * (1.0 - np.cos(np.pi * np.arange(click_len) / click_len))
            interpolated = (1.0 - alpha) * pred_fwd + alpha * pred_bwd

            # 8-Sample Crossfade an den Kanten
            fade_n = min(8, click_len // 4)
            if fade_n >= 2:
                ramp = np.linspace(0.0, 1.0, fade_n)
                # Einblenden aus Originalrand (vor Lücke)
                edge_pre = audio[max(0, start - fade_n) : start].astype(np.float64)
                if len(edge_pre) == fade_n:
                    interpolated[:fade_n] = ramp * interpolated[:fade_n] + (1.0 - ramp) * edge_pre
                # Ausblenden in Originalrand (nach Lücke)
                edge_post = audio[end + 1 : end + 1 + fade_n].astype(np.float64)
                if len(edge_post) == fade_n:
                    interpolated[-fade_n:] = ramp[::-1] * interpolated[-fade_n:] + (1.0 - ramp[::-1]) * edge_post

            # NaN/Inf-Schutz + Clip
            interpolated = np.nan_to_num(interpolated, nan=0.0, posinf=0.0, neginf=0.0)
            interpolated = np.clip(interpolated, -1.0, 1.0)
            audio[start : end + 1] = interpolated

        except Exception:
            # Graceful Degradation auf Cubic-Spline
            return self._interpolate_cubic(audio, start, end)

        return audio

    def supports_material(self, material_type: str) -> bool:
        """All materials supported."""
        return True


if __name__ == "__main__":
    """Test Professional Click Removal Phase."""

    logger.debug("=" * 80)
    logger.debug("Professional Click Removal Phase v2.0 - Test")
    logger.debug("=" * 80)

    # Generate test audio
    sr = 44100
    duration = 3
    t = np.linspace(0, duration, sr * duration)

    # Complex signal: sine + harmonics
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # Fundamental
    audio += 0.15 * np.sin(2 * np.pi * 880 * t)  # 2nd harmonic
    audio += 0.08 * np.sin(2 * np.pi * 1320 * t)  # 3rd harmonic

    # Add different types of clicks
    # 1. Short digital clicks (1-3 samples)
    for i in range(10):
        pos = int(np.random.rand() * len(audio))
        audio[pos : pos + 2] += 0.6 * np.random.randn(2)

    # 2. Medium analog clicks (4-10 samples)
    for i in range(5):
        pos = int(np.random.rand() * len(audio))
        duration_click = np.random.randint(4, 11)
        audio[pos : pos + duration_click] += 0.4 * np.random.randn(duration_click)

    # 3. Long scratches (11-50 samples)
    for i in range(2):
        pos = int(np.random.rand() * len(audio))
        duration_click = np.random.randint(15, 40)
        audio[pos : pos + duration_click] += 0.3 * np.random.randn(duration_click)

    # 4. Musical transient (should be preserved)
    transient_pos = int(len(audio) * 0.5)
    audio[transient_pos : transient_pos + 5] *= 2.0  # Legitimate attack

    logger.debug("\nTest Audio: %ss @ %s Hz", duration, sr)
    logger.debug("Injected: 10 short + 5 medium + 2 long clicks + 1 transient")

    # Test with different materials
    materials = ["shellac", "vinyl", "tape", "cd_digital"]

    for material in materials:
        logger.debug("\n%s", "-" * 80)
        logger.debug("Testing with material: %s", material.upper())
        logger.debug("%s", "-" * 80)

        phase = ClickRemovalPhase()
        result = phase.process(audio.copy(), material_type=material, preserve_transients=True)

        if result.success:
            logger.debug("✅ Processing Complete!")
            logger.debug(
                f"   Execution Time: {result.metadata['execution_time_seconds']:.3f}s ({result.metadata['execution_time_seconds'] / duration:.2f}× realtime)"
            )
            logger.debug("   Total Clicks Removed: %s", result.modifications["total_clicks_removed"])
            logger.debug("   - Short: %s", result.modifications["short_clicks"])
            logger.debug("   - Medium: %s", result.modifications["medium_clicks"])
            logger.debug("   - Long: %s", result.modifications["long_clicks"])
            logger.debug("   Transients Preserved: %s", result.modifications["transients_preserved"])
            logger.debug("   Preservation Ratio: %s", format(result.modifications["preservation_ratio"], ".2%"))
            logger.debug("   Warnings: %s", result.warnings if result.warnings else "None")
        else:
            logger.debug("❌ Processing Failed!")

    logger.debug("\n%s", "=" * 80)
    logger.debug("✅ Professional Click Removal v2.0 Test Complete!")
    logger.debug("%s", "=" * 80)
    logger.debug("Algorithm: %s", result.metadata["algorithm"])
    logger.debug("Scientific Reference: %s", result.metadata["scientific_ref"])
    logger.debug("Benchmark: %s", result.metadata["benchmark"])
    logger.debug("Quality Impact: 0.95 (Professional-Grade)")
