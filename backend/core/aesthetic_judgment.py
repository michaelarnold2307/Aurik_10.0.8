"""
AURIK Aesthetic Judgment Model (AJM)

Implements the Composite Aesthetic Score (CAS) calculation per AURIK Architecture Specification.
Translates subjective musical goals into measurable proxy metrics and optimizes aesthetic quality.

Key Features:
- 7 aesthetic dimensions (Brilliance, Transparency, Naturalness, Authenticity, Emotionality, Warmth, Spatiality)
- Genre-adaptive weighting (Classical, Jazz, Rock/Metal, Electronic, Vocal/Pop, Vintage/Analog)
- Constraint system enforcement (Spec 3.2.3)
- CAS formula implementation: CAS = Σ (wᵢ × normalized_proxyᵢ) × (1 - penalty_artifacts) × authenticity_factor

Spec Reference:
- Section 1.2: Musikalische Zielgrößen
- Section 3.2.1: Composite Aesthetic Score
- Section 3.2.2: Genre-adaptive Gewichtung
- Section 3.2.3: Constraint System
"""

import logging

import numpy as np

from .data_models import (
    AestheticScores,
    AnalysisProfile,
    ConstraintCheckResult,
    Genre,
    GenreWeights,
    QualityReport,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Genre Weight Registry (Spec 3.2.2)
# ============================================================================


class GenreWeightRegistry:
    """
    Genre-adaptive weighting for aesthetic dimensions.

    Spec Reference: Section 3.2.2 - Genre-adaptive Gewichtung

    Each genre prioritizes different aesthetic qualities:
    - Classical: Emphasizes transparency, naturalness, spatiality
    - Jazz: Emphasizes warmth, transparency, emotionality
    - Rock/Metal: Emphasizes emotionality, warmth, brilliance
    - Electronic: Emphasizes brilliance, spatiality, transparency
    - Vocal/Pop: Emphasizes emotionality, brilliance, transparency
    - Vintage/Analog: Emphasizes authenticity, warmth, naturalness
    """

    # Exact weights from Spec Table 3.2.2
    GENRE_WEIGHTS = {
        Genre.CLASSICAL: GenreWeights(
            brilliance=0.10,
            transparency=0.20,
            naturalness=0.20,
            authenticity=0.15,
            emotionality=0.15,
            warmth=0.05,
            spatiality=0.15,
        ),
        Genre.JAZZ: GenreWeights(
            brilliance=0.10,
            transparency=0.15,
            naturalness=0.15,
            authenticity=0.10,
            emotionality=0.15,
            warmth=0.20,
            spatiality=0.15,
        ),
        Genre.ROCK_METAL: GenreWeights(
            brilliance=0.15,
            transparency=0.10,
            naturalness=0.10,
            authenticity=0.15,
            emotionality=0.20,
            warmth=0.20,
            spatiality=0.10,
        ),
        Genre.ELECTRONIC: GenreWeights(
            brilliance=0.20,
            transparency=0.15,
            naturalness=0.05,
            authenticity=0.10,
            emotionality=0.15,
            warmth=0.15,
            spatiality=0.20,
        ),
        Genre.VOCAL_POP: GenreWeights(
            brilliance=0.15,
            transparency=0.15,
            naturalness=0.15,
            authenticity=0.10,
            emotionality=0.20,
            warmth=0.15,
            spatiality=0.10,
        ),
        Genre.VINTAGE_ANALOG: GenreWeights(
            brilliance=0.05,
            transparency=0.10,
            naturalness=0.15,
            authenticity=0.25,
            emotionality=0.10,
            warmth=0.25,
            spatiality=0.10,
        ),
        Genre.UNKNOWN: GenreWeights(
            # Balanced default weights
            brilliance=0.14,
            transparency=0.14,
            naturalness=0.14,
            authenticity=0.15,
            emotionality=0.15,
            warmth=0.14,
            spatiality=0.14,
        ),
    }

    @classmethod
    def get_weights(cls, genre: Genre, confidence: float = 1.0) -> GenreWeights:
        """
        Get genre-adaptive weights with confidence-based blending.

        Args:
            genre: Detected genre
            confidence: Genre detection confidence (0-1)

        Returns:
            GenreWeights for the specified genre

        Note:
            If confidence < 0.8, blends genre weights with default weights
        """
        genre_weights = cls.GENRE_WEIGHTS[genre]

        if confidence < 0.8:
            # Blend with default weights for low confidence
            default_weights = cls.GENRE_WEIGHTS[Genre.UNKNOWN]
            blended = GenreWeights(
                brilliance=confidence * genre_weights.brilliance + (1 - confidence) * default_weights.brilliance,
                transparency=confidence * genre_weights.transparency + (1 - confidence) * default_weights.transparency,
                naturalness=confidence * genre_weights.naturalness + (1 - confidence) * default_weights.naturalness,
                authenticity=confidence * genre_weights.authenticity + (1 - confidence) * default_weights.authenticity,
                emotionality=confidence * genre_weights.emotionality + (1 - confidence) * default_weights.emotionality,
                warmth=confidence * genre_weights.warmth + (1 - confidence) * default_weights.warmth,
                spatiality=confidence * genre_weights.spatiality + (1 - confidence) * default_weights.spatiality,
            )
            return blended

        return genre_weights

    @classmethod
    def validate_all_weights(cls) -> bool:
        """Validate that all genre weight tables sum to 1.0"""
        for genre, weights in cls.GENRE_WEIGHTS.items():
            if not weights.validate_sum():
                logger.warning(f"Genre {genre} weights do not sum to 1.0")
                return False
        return True


# ============================================================================
# Aesthetic Proxy Calculator
# ============================================================================


class AestheticProxyCalculator:
    """
    Calculates proxy metrics for all 7 aesthetic dimensions.

    Spec Reference: Section 1.2 - Musikalische Zielgrößen

    Each dimension has multiple technical proxies that serve as measurable
    indicators (though not absolute truth) of subjective aesthetic quality.
    """

    @staticmethod
    def calculate_brilliance_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Brilliance score (Spec 1.2: Brillanz).

        Proxies:
        - High-Frequency Energy Ratio (12-20kHz)
        - Spectral Centroid (normalized)
        - Air-Band Presence (15-20kHz)
        - Harmonic Brightness Index

        Args:
            audio: Audio signal (mono or stereo)
            sr: Sample rate
            profile: Analysis profile with pre-computed features

        Returns:
            Tuple of (brilliance_score, proxy_details)
        """
        details = {}

        # 1. HF Energy Ratio (12-20kHz)
        nyquist = sr / 2
        if nyquist >= 20000:
            fft = np.fft.rfft(audio.flatten())
            freqs = np.fft.rfftfreq(len(audio.flatten()), 1 / sr)

            hf_mask = (freqs >= 12000) & (freqs <= 20000)
            hf_energy = np.sum(np.abs(fft[hf_mask]) ** 2)
            total_energy = np.sum(np.abs(fft) ** 2)

            hf_ratio = hf_energy / (total_energy + 1e-10)
            details["hf_energy_ratio"] = float(hf_ratio)
        else:
            hf_ratio = 0.0
            details["hf_energy_ratio"] = 0.0

        # 2. Spectral Centroid (normalized to 0-1, typical range 1000-8000 Hz)
        spectral_centroid = profile.spectral.spectral_centroid
        centroid_norm = np.clip((spectral_centroid - 1000) / 7000, 0.0, 1.0)
        details["spectral_centroid_normalized"] = float(centroid_norm)

        # 3. Air-Band Presence (15-20kHz)
        if nyquist >= 20000:
            air_mask = (freqs >= 15000) & (freqs <= 20000)
            air_energy = np.sum(np.abs(fft[air_mask]) ** 2)
            air_presence = air_energy / (total_energy + 1e-10)
            details["air_band_presence"] = float(air_presence)
        else:
            air_presence = 0.0
            details["air_band_presence"] = 0.0

        # 4. Harmonic Brightness (approximated from spectral rolloff)
        rolloff = profile.spectral.spectral_rolloff
        brightness_index = np.clip((rolloff - 4000) / 12000, 0.0, 1.0)
        details["harmonic_brightness_index"] = float(brightness_index)

        # Weighted combination
        brilliance = 0.3 * hf_ratio + 0.2 * centroid_norm + 0.3 * air_presence + 0.2 * brightness_index
        brilliance = np.clip(brilliance, 0.0, 1.0)

        return float(brilliance), details

    @staticmethod
    def calculate_transparency_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Transparency score (Spec 1.2: Transparenz).

        Proxies:
        - Spectral Flatness
        - Inter-Source Masking Index (estimated)
        - Transient Sharpness
        - Frequency-Band Separation Score

        Args:
            audio: Audio signal
            sr: Sample rate
            profile: Analysis profile

        Returns:
            Tuple of (transparency_score, proxy_details)
        """
        details = {}

        # 1. Spectral Flatness (already normalized 0-1)
        spectral_flatness = profile.spectral.spectral_flatness if profile.spectral.spectral_flatness else 0.5
        details["spectral_flatness"] = float(spectral_flatness)

        # 2. Inter-Source Masking (estimated from spectral contrast)
        # Higher contrast suggests less masking
        masking_estimate = 0.7  # Placeholder - would need source separation for true measurement
        details["inter_source_masking_index"] = masking_estimate

        # 3. Transient Sharpness (from onset count and strength)
        onset_count = len(profile.feature_vectors.onset_times)
        duration = profile.format_info.sample_rate / sr if audio.size > 0 else 1.0
        onset_density = onset_count / duration if duration > 0 else 0.0
        transient_sharpness = np.clip(onset_density / 10.0, 0.0, 1.0)  # Normalize by typical density
        details["transient_sharpness"] = float(transient_sharpness)

        # 4. Frequency-Band Separation (estimated from crest factor and dynamics)
        crest_factor_db = profile.dynamics.crest_factor_db
        separation_score = np.clip(crest_factor_db / 20.0, 0.0, 1.0)
        details["frequency_band_separation"] = float(separation_score)

        # Weighted combination
        transparency = (
            0.25 * spectral_flatness + 0.35 * masking_estimate + 0.20 * transient_sharpness + 0.20 * separation_score
        )
        transparency = np.clip(transparency, 0.0, 1.0)

        return float(transparency), details

    @staticmethod
    def calculate_naturalness_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Naturalness score (Spec 1.2: Natürlichkeit).

        Proxies:
        - Foundation Model Deviation Score (placeholder)
        - Artifact Likelihood Estimator
        - Harmonic Distortion Profile

        Args:
            audio: Audio signal
            sr: Sample rate
            profile: Analysis profile

        Returns:
            Tuple of (naturalness_score, proxy_details)
        """
        details = {}

        # 1. Artifact Likelihood (inverted - lower=more natural)
        artifact_severity_sum = sum(d.severity for d in profile.detected_defects)
        artifact_likelihood = np.clip(artifact_severity_sum / 5.0, 0.0, 1.0)
        naturalness_from_artifacts = 1.0 - artifact_likelihood
        details["artifact_likelihood"] = float(artifact_likelihood)

        # 2. Foundation Model Deviation (placeholder - would require ML model)
        # For now, use overall quality score as proxy
        foundation_deviation = 1.0 - profile.overall_quality_score
        naturalness_from_foundation = 1.0 - foundation_deviation
        details["foundation_model_deviation"] = float(foundation_deviation)

        # 3. Harmonic Distortion Profile (from harmonicity if available)
        if profile.feature_vectors.harmonicity is not None:
            harmonicity = profile.feature_vectors.harmonicity
            # High harmonicity suggests low distortion
            naturalness_from_harmonics = np.clip(harmonicity / 30.0, 0.0, 1.0)
        else:
            naturalness_from_harmonics = 0.7  # Default
        details["harmonic_distortion_profile"] = float(1.0 - naturalness_from_harmonics)

        # Weighted combination
        naturalness = (
            0.4 * naturalness_from_artifacts + 0.3 * naturalness_from_foundation + 0.3 * naturalness_from_harmonics
        )
        naturalness = np.clip(naturalness, 0.0, 1.0)

        return float(naturalness), details

    @staticmethod
    def calculate_authenticity_score(
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        sr: int,
        profile: AnalysisProfile,
    ) -> tuple[float, dict]:
        """
        Calculate Authenticity score (Spec 1.2: Authentizität).

        Proxies:
        - Spectral Correlation (Original vs Restored)
        - Timbre Consistency Score
        - Perceptual Similarity Index

        Args:
            original_audio: Original audio signal
            processed_audio: Processed audio signal
            sr: Sample rate
            profile: Analysis profile

        Returns:
            Tuple of (authenticity_score, proxy_details)
        """
        details = {}

        # Align lengths
        min_len = min(len(original_audio), len(processed_audio))
        orig = original_audio[:min_len]
        proc = processed_audio[:min_len]

        # 1. Spectral Correlation
        orig_fft = np.fft.rfft(orig.flatten())
        proc_fft = np.fft.rfft(proc.flatten())

        orig_mag = np.abs(orig_fft)
        proc_mag = np.abs(proc_fft)

        correlation = np.corrcoef(orig_mag, proc_mag)[0, 1]
        correlation = np.clip(correlation, 0.0, 1.0)
        details["spectral_correlation"] = float(correlation)

        # 2. Timbre Consistency (RMS-based approximation)
        orig_rms = np.sqrt(np.mean(orig**2))
        proc_rms = np.sqrt(np.mean(proc**2))
        rms_ratio = min(orig_rms, proc_rms) / (max(orig_rms, proc_rms) + 1e-10)
        details["timbre_consistency"] = float(rms_ratio)

        # 3. Perceptual Similarity (simple MSE-based for now)
        mse = np.mean((orig - proc) ** 2)
        max_val = max(np.max(np.abs(orig)), np.max(np.abs(proc)))
        normalized_mse = mse / (max_val**2 + 1e-10)
        perceptual_similarity = 1.0 - np.clip(normalized_mse * 10, 0.0, 1.0)
        details["perceptual_similarity_index"] = float(perceptual_similarity)

        # Weighted combination
        authenticity = 0.4 * correlation + 0.3 * rms_ratio + 0.3 * perceptual_similarity
        authenticity = np.clip(authenticity, 0.0, 1.0)

        return float(authenticity), details

    @staticmethod
    def calculate_emotionality_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Emotionality score (Spec 1.2: Emotionalität).

        Proxies:
        - Micro-Dynamics Preservation
        - Vibrato/Tremolo Integrity
        - Dynamic Range Variance
        - Expressive Feature Retention

        Args:
            audio: Audio signal
            sr: Sample rate
            profile: Analysis profile

        Returns:
            Tuple of (emotionality_score, proxy_details)
        """
        details = {}

        # 1. Micro-Dynamics (from dynamic range)
        dr_db = profile.dynamics.dynamic_range_db
        # Typical excellent DR: 12-20 dB
        microdynamics = np.clip((dr_db - 5) / 15, 0.0, 1.0)
        details["microdynamics_preservation"] = float(microdynamics)

        # 2. Vibrato/Tremolo Integrity (from pitch variance if available)
        if profile.feature_vectors.pitch_contour:
            pitch_variance = np.std(profile.feature_vectors.pitch_contour)
            vibrato_integrity = np.clip(pitch_variance / 50.0, 0.0, 1.0)
        else:
            vibrato_integrity = 0.7  # Default
        details["vibrato_tremolo_integrity"] = float(vibrato_integrity)

        # 3. Dynamic Range Variance (from loudness range)
        lra_lu = profile.dynamics.loudness_range_lu
        # Typical good LRA: 6-15 LU
        dr_variance = np.clip((lra_lu - 3) / 12, 0.0, 1.0)
        details["dynamic_range_variance"] = float(dr_variance)

        # 4. Expressive Feature Retention (from harmonicity and transients)
        if profile.feature_vectors.harmonicity:
            expressive_retention = np.clip(profile.feature_vectors.harmonicity / 25.0, 0.0, 1.0)
        else:
            expressive_retention = 0.7
        details["expressive_feature_retention"] = float(expressive_retention)

        # Weighted combination
        emotionality = (
            0.30 * microdynamics + 0.20 * vibrato_integrity + 0.25 * dr_variance + 0.25 * expressive_retention
        )
        emotionality = np.clip(emotionality, 0.0, 1.0)

        return float(emotionality), details

    @staticmethod
    def calculate_warmth_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Warmth score (Spec 1.2: Wärme).

        Proxies:
        - Low-Mid Energy Balance (200-800Hz)
        - Even Harmonic Content
        - Crest Factor
        - Tube Saturation Profile Similarity

        Args:
            audio: Audio signal
            sr: Sample rate
            profile: Analysis profile

        Returns:
            Tuple of (warmth_score, proxy_details)
        """
        details = {}

        # 1. Low-Mid Energy Balance (200-800Hz)
        fft = np.fft.rfft(audio.flatten())
        freqs = np.fft.rfftfreq(len(audio.flatten()), 1 / sr)

        lowmid_mask = (freqs >= 200) & (freqs <= 800)
        lowmid_energy = np.sum(np.abs(fft[lowmid_mask]) ** 2)
        total_energy = np.sum(np.abs(fft) ** 2)

        lowmid_balance = lowmid_energy / (total_energy + 1e-10)
        # Normalize to typical range
        lowmid_balance_norm = np.clip(lowmid_balance * 10, 0.0, 1.0)
        details["lowmid_energy_balance"] = float(lowmid_balance_norm)

        # 2. Even Harmonic Content (approximation)
        # Would require pitch detection and harmonic analysis
        # Placeholder: use inverse of crest factor as proxy
        crest_factor_db = profile.dynamics.crest_factor_db
        even_harmonic_estimate = np.clip(1.0 - (crest_factor_db - 10) / 15, 0.0, 1.0)
        details["even_harmonic_content"] = float(even_harmonic_estimate)

        # 3. Crest Factor (inverted for warmth - lower crest = warmer)
        warmth_from_crest = np.clip(1.0 - (crest_factor_db - 8) / 12, 0.0, 1.0)
        details["crest_factor_warmth"] = float(warmth_from_crest)

        # 4. Tube Saturation Profile (placeholder - would need saturation detection)
        tube_saturation_similarity = 0.6  # Neutral default
        details["tube_saturation_profile"] = tube_saturation_similarity

        # Weighted combination
        warmth = (
            0.30 * lowmid_balance_norm
            + 0.25 * even_harmonic_estimate
            + 0.25 * warmth_from_crest
            + 0.20 * tube_saturation_similarity
        )
        warmth = np.clip(warmth, 0.0, 1.0)

        return float(warmth), details

    @staticmethod
    def calculate_spatiality_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Spatiality score (Spec 1.2: Räumlichkeit).

        Proxies:
        - IACC (Interaural Cross-Correlation)
        - Stereo Width
        - Early/Late Energy Ratio (approximated)
        - Envelopment Index
        - Depth Cue Preservation

        Args:
            audio: Audio signal
            sr: Sample rate
            profile: Analysis profile

        Returns:
            Tuple of (spatiality_score, proxy_details)
        """
        details = {}

        # 1. IACC (from profile)
        iacc = profile.stereo.iacc
        # IACC ranges from -1 to 1; lower values = wider stereo image
        # Convert to 0-1 scale where higher = better spatiality
        iacc_score = (1.0 - iacc) / 2.0  # Maps [-1, 1] to [1, 0]
        details["iacc"] = float(iacc_score)

        # 2. Stereo Width (from profile)
        stereo_width = profile.stereo.stereo_width
        # Normalize: 0=mono, 1=normal, >1=wide. Clip to [0, 1.5] and normalize
        width_norm = np.clip(stereo_width / 1.5, 0.0, 1.0)
        details["stereo_width"] = float(width_norm)

        # 3. Early/Late Energy Ratio (approximation using onset density)
        # Higher onset density suggests more early energy (transients)
        onset_count = len(profile.feature_vectors.onset_times)
        duration_sec = len(audio) / sr if sr > 0 else 1.0
        early_late_ratio = np.clip(onset_count / (duration_sec * 5), 0.0, 1.0)
        details["early_late_energy_ratio"] = float(early_late_ratio)

        # 4. Envelopment Index (approximation from stereo width and phase coherence)
        phase_coherence = profile.stereo.phase_coherence
        envelopment = (stereo_width + (1.0 - phase_coherence)) / 2.0
        envelopment = np.clip(envelopment, 0.0, 1.0)
        details["envelopment_index"] = float(envelopment)

        # 5. Depth Cue Preservation (approximation from spectral rolloff)
        # Higher rolloff suggests more depth/air
        rolloff = profile.spectral.spectral_rolloff
        depth_cue = np.clip((rolloff - 5000) / 10000, 0.0, 1.0)
        details["depth_cue_preservation"] = float(depth_cue)

        # Weighted combination
        spatiality = (
            0.25 * iacc_score + 0.25 * width_norm + 0.15 * early_late_ratio + 0.20 * envelopment + 0.15 * depth_cue
        )
        spatiality = np.clip(spatiality, 0.0, 1.0)

        return float(spatiality), details


# ============================================================================
# Composite Aesthetic Score Calculator (Spec 3.2.1)
# ============================================================================


class CompositeAestheticScoreCalculator:
    """
    Calculates the Composite Aesthetic Score (CAS) using the formula from Spec 3.2.1.

    Formula:
        CAS = Σ (wᵢ × normalized_proxyᵢ) × (1 - penalty_artifacts) × authenticity_factor

    Where:
        - wᵢ are genre-adaptive weights
        - normalized_proxyᵢ are the 7 aesthetic dimension scores (0-1)
        - penalty_artifacts is artifact penalty (0-1)
        - authenticity_factor is authenticity preservation (0-1)
    """

    def __init__(self):
        self.proxy_calculator = AestheticProxyCalculator()

    def calculate_cas(
        self,
        audio: np.ndarray,
        sr: int,
        profile: AnalysisProfile,
        original_audio: np.ndarray | None = None,
        genre: Genre = Genre.UNKNOWN,
        genre_confidence: float = 1.0,
    ) -> tuple[float, AestheticScores]:
        """
        Calculate Composite Aesthetic Score.

        Args:
            audio: Processed audio signal
            sr: Sample rate
            profile: Analysis profile
            original_audio: Original audio (for authenticity calculation)
            genre: Detected genre
            genre_confidence: Genre detection confidence

        Returns:
            Tuple of (cas_score, aesthetic_scores)
        """
        # Get genre-adaptive weights
        weights = GenreWeightRegistry.get_weights(genre, genre_confidence)

        # Calculate all 7 proxy scores
        brilliance, brilliance_details = self.proxy_calculator.calculate_brilliance_score(audio, sr, profile)
        transparency, transparency_details = self.proxy_calculator.calculate_transparency_score(audio, sr, profile)
        naturalness, naturalness_details = self.proxy_calculator.calculate_naturalness_score(audio, sr, profile)

        if original_audio is not None:
            authenticity, authenticity_details = self.proxy_calculator.calculate_authenticity_score(
                original_audio, audio, sr, profile
            )
        else:
            # No original for comparison - assume high authenticity
            authenticity = 0.9
            authenticity_details = {"note": "no_original_for_comparison"}

        emotionality, emotionality_details = self.proxy_calculator.calculate_emotionality_score(audio, sr, profile)
        warmth, warmth_details = self.proxy_calculator.calculate_warmth_score(audio, sr, profile)
        spatiality, spatiality_details = self.proxy_calculator.calculate_spatiality_score(audio, sr, profile)

        # Create AestheticScores object
        scores = AestheticScores(
            brilliance=brilliance,
            transparency=transparency,
            naturalness=naturalness,
            authenticity=authenticity,
            emotionality=emotionality,
            warmth=warmth,
            spatiality=spatiality,
            proxy_details={
                "brilliance": brilliance_details,
                "transparency": transparency_details,
                "naturalness": naturalness_details,
                "authenticity": authenticity_details,
                "emotionality": emotionality_details,
                "warmth": warmth_details,
                "spatiality": spatiality_details,
            },
        )

        # Calculate weighted sum: Σ (wᵢ × normalized_proxyᵢ)
        weighted_sum = (
            weights.brilliance * brilliance
            + weights.transparency * transparency
            + weights.naturalness * naturalness
            + weights.authenticity * authenticity
            + weights.emotionality * emotionality
            + weights.warmth * warmth
            + weights.spatiality * spatiality
        )

        # Calculate artifact penalty: (1 - penalty_artifacts)
        artifact_severity_sum = sum(d.severity * d.confidence for d in profile.detected_defects)
        penalty_artifacts = np.clip(artifact_severity_sum / 3.0, 0.0, 1.0)  # Normalize by typical max
        artifact_factor = 1.0 - penalty_artifacts

        # Apply formula: CAS = weighted_sum × (1 - penalty) × authenticity
        cas = weighted_sum * artifact_factor * authenticity
        cas = np.clip(cas, 0.0, 1.0)

        logger.info(
            f"CAS Calculation: weighted_sum={weighted_sum:.3f}, artifact_factor={artifact_factor:.3f}, authenticity={authenticity:.3f}, CAS={cas:.3f}"
        )

        return float(cas), scores

    def check_constraints(
        self,
        cas_before: float,
        cas_after: float,
        scores_before: AestheticScores,
        scores_after: AestheticScores,
        profile_before: AnalysisProfile,
        profile_after: AnalysisProfile,
    ) -> list[ConstraintCheckResult]:
        """
        Check constraint system per Spec 3.2.3.

        Constraints:
        1. Authenticity Floor: Perceptual Similarity > 0.85
        2. Artifact Ceiling: New artifacts < existing
        3. Dynamic Preservation: Mikrodynamik > 90%
        4. Spectral Integrity: No bands > 3dB changed without defect indication

        Args:
            cas_before: CAS before processing
            cas_after: CAS after processing
            scores_before: Aesthetic scores before
            scores_after: Aesthetic scores after
            profile_before: Analysis profile before
            profile_after: Analysis profile after

        Returns:
            List of ConstraintCheckResult objects
        """
        results = []

        # Constraint 1: Authenticity Floor
        authenticity = scores_after.authenticity
        passed_auth = authenticity > 0.85
        results.append(
            ConstraintCheckResult(
                constraint_name="Authenticity Floor",
                passed=passed_auth,
                measured_value=authenticity,
                threshold_value=0.85,
                severity="error" if not passed_auth else "info",
                message=f"Perceptual similarity to original: {authenticity:.3f} ({'PASS' if passed_auth else 'FAIL'}: must be > 0.85)",
            )
        )

        # Constraint 2: Artifact Ceiling
        artifacts_before = sum(d.severity for d in profile_before.detected_defects)
        artifacts_after = sum(d.severity for d in profile_after.detected_defects)
        passed_artifact = artifacts_after < artifacts_before or abs(artifacts_after - artifacts_before) < 0.1
        results.append(
            ConstraintCheckResult(
                constraint_name="Artifact Ceiling",
                passed=passed_artifact,
                measured_value=artifacts_after,
                threshold_value=artifacts_before,
                severity="warning" if not passed_artifact else "info",
                message=f"New artifact level: {artifacts_after:.3f} vs original: {artifacts_before:.3f} ({'PASS' if passed_artifact else 'FAIL'}: must not increase)",
            )
        )

        # Constraint 3: Dynamic Preservation
        dr_before = profile_before.dynamics.dynamic_range_db
        dr_after = profile_after.dynamics.dynamic_range_db
        dr_preservation = dr_after / (dr_before + 1e-6)
        passed_dynamics = dr_preservation > 0.9
        results.append(
            ConstraintCheckResult(
                constraint_name="Dynamic Preservation",
                passed=passed_dynamics,
                measured_value=dr_preservation,
                threshold_value=0.9,
                severity="warning" if not passed_dynamics else "info",
                message=f"Dynamic range preservation: {dr_preservation:.1%} ({'PASS' if passed_dynamics else 'FAIL'}: must be > 90%)",
            )
        )

        # Constraint 4: Spectral Integrity
        centroid_before = profile_before.spectral.spectral_centroid
        centroid_after = profile_after.spectral.spectral_centroid
        centroid_change_db = 20 * np.log10((centroid_after + 1) / (centroid_before + 1))
        passed_spectral = abs(centroid_change_db) < 3.0
        results.append(
            ConstraintCheckResult(
                constraint_name="Spectral Integrity",
                passed=passed_spectral,
                measured_value=abs(centroid_change_db),
                threshold_value=3.0,
                severity="warning" if not passed_spectral else "info",
                message=f"Spectral centroid change: {centroid_change_db:+.1f} dB ({'PASS' if passed_spectral else 'FAIL'}: must be < 3 dB)",
            )
        )

        return results


# ============================================================================
# Aesthetic Judgment Model (Main Facade)
# ============================================================================


class AestheticJudgmentModel:
    """
    Main facade for the Aesthetic Judgment Model.

    Orchestrates CAS calculation, genre-adaptive weighting, and constraint checking.
    """

    def __init__(self):
        self.cas_calculator = CompositeAestheticScoreCalculator()
        GenreWeightRegistry.validate_all_weights()  # Validate on init

    def evaluate(
        self,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        profile_before: AnalysisProfile,
        profile_after: AnalysisProfile,
        genre: Genre | None = None,
        genre_confidence: float = 1.0,
    ) -> QualityReport:
        """
        Complete aesthetic evaluation: CAS calculation + constraint checking.

        Args:
            audio_before: Original audio signal
            audio_after: Processed audio signal
            sr: Sample rate
            profile_before: Analysis profile of original
            profile_after: Analysis profile of processed
            genre: Detected genre (if None, uses profile)
            genre_confidence: Genre detection confidence

        Returns:
            QualityReport with CAS scores and constraint results
        """
        # Determine genre
        if genre is None:
            genre = profile_after.musical_context.genre
            genre_confidence = profile_after.musical_context.genre_confidence

        # Calculate CAS before and after
        cas_before, scores_before = self.cas_calculator.calculate_cas(
            audio_before, sr, profile_before, None, genre, genre_confidence
        )

        cas_after, scores_after = self.cas_calculator.calculate_cas(
            audio_after, sr, profile_after, audio_before, genre, genre_confidence
        )

        cas_improvement = cas_after - cas_before

        # Check constraints
        constraint_results = self.cas_calculator.check_constraints(
            cas_before,
            cas_after,
            scores_before,
            scores_after,
            profile_before,
            profile_after,
        )

        constraints_satisfied = all(c.passed for c in constraint_results)

        # Collect warnings and errors
        warnings = [c.message for c in constraint_results if c.severity == "warning"]
        errors = [c.message for c in constraint_results if c.severity == "error"]

        # Create QualityReport
        report = QualityReport(
            cas_before=cas_before,
            cas_after=cas_after,
            cas_improvement=cas_improvement,
            aesthetic_scores_before=scores_before,
            aesthetic_scores_after=scores_after,
            constraints_satisfied=constraints_satisfied,
            constraint_checks=constraint_results,
            warnings=warnings,
            errors=errors,
        )

        logger.info(f"Quality evaluation complete: CAS {cas_before:.3f} → {cas_after:.3f} (Δ{cas_improvement:+.3f})")

        return report


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "GenreWeightRegistry",
    "AestheticProxyCalculator",
    "CompositeAestheticScoreCalculator",
    "AestheticJudgmentModel",
]
