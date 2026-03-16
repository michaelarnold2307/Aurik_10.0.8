"""
Treatment Recommender
=====================

Recommends treatments for detected audio defects.
Maps defects to DSP modules with optimal parameters.
"""

from typing import Any

from backend.defect_detection.base import DefectInstance, DefectType, SeverityLevel, TreatmentRecommendation


class TreatmentRecommender:
    """
    Recommends treatments for detected audio defects.

    Maps defect type + severity to appropriate DSP module + parameters.
    Similar to iZotope RX10's "Repair Assistant".
    """

    def __init__(self):
        # Treatment mappings: defect_type -> (module_path, method_name)
        self.treatment_map = {
            DefectType.CLIPPING: ("dsp.automatic_declipper", "declip"),
            DefectType.CLICKS: ("dsp.automatic_declicker", "declick"),
            DefectType.CRACKLE: ("dsp.automatic_decrackler", "decrackle"),
            DefectType.BROADBAND_NOISE: ("dsp.automatic_denoiser", "denoise"),
            DefectType.HUM: ("dsp.automatic_dehum", "dehum"),
            DefectType.BUZZ: ("dsp.automatic_debuzzer", "debuzz"),
            DefectType.DISTORTION: ("dsp.harmonic_exciter", "reduce_distortion"),
            DefectType.RUMBLE: ("dsp.rumble_filter", "filter_rumble"),
            DefectType.HF_ROLLOFF: ("dsp.bandwidth_extender", "extend_bandwidth"),
            DefectType.STEREO_IMBALANCE: ("dsp.stereo_image_correction", "correct_stereo"),
            DefectType.DC_OFFSET: ("dsp.classic_filters", "remove_dc_offset"),
        }

    def recommend(self, defect: DefectInstance) -> TreatmentRecommendation:
        """
        Recommend treatment for a defect.

        Args:
            defect: Detected defect instance

        Returns:
            Treatment recommendation with method, params, priority
        """
        if defect.type not in self.treatment_map:
            return self._create_no_treatment(defect)

        module_path, method = self.treatment_map[defect.type]

        # Generate parameters based on defect severity
        params = self._generate_params(defect)

        # Determine priority (1=highest, 5=lowest)
        priority = self._calculate_priority(defect)

        # Estimate improvement
        expected_improvement = self._estimate_improvement(defect)

        # List potential side effects
        side_effects = self._list_side_effects(defect.type, defect.severity)

        # Check if manual verification needed
        requires_manual = self._requires_manual_check(defect)

        return TreatmentRecommendation(
            method=method,
            module_path=module_path,
            params=params,
            priority=priority,
            expected_improvement=expected_improvement,
            side_effects=side_effects,
            requires_manual_check=requires_manual,
        )

    def recommend_batch(self, defects: list[DefectInstance]) -> list[TreatmentRecommendation]:
        """
        Recommend treatments for multiple defects.

        Returns treatments sorted by priority.
        """
        treatments = [self.recommend(d) for d in defects]

        # Sort by priority (lower number = higher priority)
        treatments.sort(key=lambda t: t.priority)

        # Remove duplicates (same method)
        seen_methods = set()
        unique_treatments = []
        for t in treatments:
            if t.method not in seen_methods:
                unique_treatments.append(t)
                seen_methods.add(t.method)

        return unique_treatments

    def _generate_params(self, defect: DefectInstance) -> dict[str, Any]:
        """Generate treatment parameters based on defect severity."""
        severity = defect.severity

        if defect.type == DefectType.CLIPPING:
            return {
                "strength": min(0.3 + severity * 0.7, 1.0),  # 0.3 - 1.0
                "iterations": int(1 + severity * 4),  # 1 - 5
                "window_size": 2048 if severity < 0.5 else 4096,
            }

        elif defect.type == DefectType.CLICKS:
            return {
                "threshold": max(0.5 - severity * 0.4, 0.1),  # 0.5 - 0.1 (lower = more aggressive)
                "window_size": int(32 + severity * 96),  # 32 - 128 samples
                "sensitivity": min(0.5 + severity * 0.5, 1.0),
            }

        elif defect.type == DefectType.CRACKLE:
            return {
                "threshold": max(0.6 - severity * 0.4, 0.2),
                "window_size": 64,
                "attack": 1.0,  # ms
                "release": 10.0,  # ms
            }

        elif defect.type == DefectType.BROADBAND_NOISE:
            return {
                "reduction_db": min(6.0 + severity * 24.0, 30.0),  # 6-30 dB
                "noise_floor_db": -60.0 + severity * 20.0,  # -60 to -40 dB
                "sensitivity": min(0.5 + severity * 0.5, 1.0),
            }

        elif defect.type == DefectType.HUM:
            freqs = defect.metrics.get("frequencies", [50.0, 60.0])
            return {
                "frequencies": freqs,
                "q_factor": 30.0,  # Narrow notch
                "num_harmonics": int(2 + severity * 6),  # 2-8 harmonics
            }

        elif defect.type == DefectType.BUZZ:
            return {
                "frequency_range": (80, 300),
                "reduction_db": min(6.0 + severity * 18.0, 24.0),
            }

        elif defect.type == DefectType.RUMBLE:
            return {
                "cutoff_hz": min(40.0 + severity * 60.0, 100.0),  # 40-100 Hz
                "order": int(2 + severity * 4),  # 2-6
                "filter_type": "highpass",
            }

        elif defect.type == DefectType.HF_ROLLOFF:
            rolloff_freq = defect.metrics.get("rolloff_frequency", 12000)
            return {
                "target_frequency": min(rolloff_freq * 1.5, 20000),
                "gain_db": min(3.0 + severity * 9.0, 12.0),
                "slope": "gentle",
            }

        elif defect.type == DefectType.STEREO_IMBALANCE:
            imbalance = defect.metrics.get("imbalance_db", 0.0)
            return {
                "correction_db": imbalance,
                "affected_channel": defect.affected_channels[0] if defect.affected_channels else 0,
            }

        elif defect.type == DefectType.DC_OFFSET:
            offset = defect.metrics.get("dc_offset", 0.0)
            return {
                "offset": offset,
                "use_highpass": True,
                "highpass_cutoff": 5.0,  # Hz
            }

        else:
            return {}

    def _calculate_priority(self, defect: DefectInstance) -> int:
        """
        Calculate treatment priority (1=highest, 5=lowest).

        Priority rules:
        - Critical defects: priority 1
        - Severe defects: priority 2
        - Moderate defects: priority 3
        - Minor defects: priority 4
        - Clipping/Distortion always high priority regardless of severity
        """
        # Clipping and distortion are always high priority
        if defect.type in (DefectType.CLIPPING, DefectType.DISTORTION):
            return 1

        # DC offset should be fixed early
        if defect.type == DefectType.DC_OFFSET:
            return 1

        # Otherwise, prioritize by severity
        if defect.severity_level == SeverityLevel.CRITICAL:
            return 1
        elif defect.severity_level == SeverityLevel.SEVERE:
            return 2
        elif defect.severity_level == SeverityLevel.MODERATE:
            return 3
        elif defect.severity_level == SeverityLevel.MINOR:
            return 4
        else:
            return 5

    def _estimate_improvement(self, defect: DefectInstance) -> float:
        """
        Estimate expected improvement (0.0 - 1.0) from treatment.

        Based on:
        - Defect type (some are easier to fix)
        - Severity (mild defects easier to fix completely)
        - Confidence (higher confidence = better estimate)
        """
        # Base improvement by defect type
        type_improvements = {
            DefectType.CLIPPING: 0.7,
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 0.8,
            DefectType.BROADBAND_NOISE: 0.75,
            DefectType.HUM: 0.95,
            DefectType.BUZZ: 0.85,
            DefectType.DISTORTION: 0.5,
            DefectType.RUMBLE: 0.9,
            DefectType.HF_ROLLOFF: 0.6,
            DefectType.STEREO_IMBALANCE: 0.95,
            DefectType.DC_OFFSET: 1.0,
        }

        base_improvement = type_improvements.get(defect.type, 0.5)

        # Adjust for severity (severe defects harder to fix completely)
        severity_factor = 1.0 - (defect.severity * 0.3)

        # Adjust for confidence
        confidence_factor = 0.7 + (defect.confidence * 0.3)

        improvement = base_improvement * severity_factor * confidence_factor

        return min(improvement, 1.0)

    def _list_side_effects(self, defect_type: DefectType, severity: float) -> list[str]:
        """List potential side effects of treatment."""
        side_effects = []

        if defect_type == DefectType.CLIPPING:
            side_effects.append("May reduce transient impact")
            if severity > 0.7:
                side_effects.append("Possible spectral smearing")

        elif defect_type == DefectType.CLICKS:
            side_effects.append("May smooth fast transients")
            if severity > 0.5:
                side_effects.append("Possible high-frequency dulling")

        elif defect_type == DefectType.BROADBAND_NOISE:
            side_effects.append("May introduce musical noise artifacts")
            side_effects.append("Possible loss of ambience")
            if severity > 0.6:
                side_effects.append("Potential spectral holes")

        elif defect_type == DefectType.HUM:
            side_effects.append("Narrow notch filters (minimal impact)")

        elif defect_type == DefectType.RUMBLE:
            side_effects.append("Reduced low-frequency energy")
            if severity > 0.5:
                side_effects.append("Possible bass thinning")

        elif defect_type == DefectType.HF_ROLLOFF:
            side_effects.append("Possible treble harshness")
            side_effects.append("May amplify noise floor")

        return side_effects

    def _requires_manual_check(self, defect: DefectInstance) -> bool:
        """Determine if manual verification recommended."""
        # High severity defects should be checked
        if defect.severity > 0.8:
            return True

        # Low confidence detections should be checked
        if defect.confidence < 0.6:
            return True

        # Distortion is subjective, needs verification
        if defect.type == DefectType.DISTORTION and defect.severity > 0.5:
            return True

        return False

    def _create_no_treatment(self, defect: DefectInstance) -> TreatmentRecommendation:
        """Create a placeholder for defects with no treatment."""
        return TreatmentRecommendation(
            method="manual_inspection",
            module_path="",
            params={},
            priority=5,
            expected_improvement=0.0,
            side_effects=["No automatic treatment available"],
            requires_manual_check=True,
        )
