"""
Perceptual Loss Functions für Aurik 8.0

Implementiert psychoakustisch und musikalisch fundierte Loss Functions:
1. PANNs-based High-Level Perceptual Loss
2. Multi-Resolution STFT Loss
3. Psychoacoustic Masking Loss
4. Musical Feature Loss (Harmonic, Rhythmic, Timbral)

Autor: Aurik Backend-Team
Version: 8.1
Datum: 14. Februar 2026
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss für verschiedene Time-Frequency Auflösungen.

    Basiert auf:
    - Yamamoto et al. (2019): "Parallel WaveGAN"
    - Défossez et al. (2020): "Real Time Speech Enhancement in the Waveform Domain"
    """

    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_lengths: list[int] | None = None,
        window: str = "hann",
        spectral_convergence_weight: float = 1.0,
        log_magnitude_weight: float = 1.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [2048, 1024, 512, 256, 128]

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes or [f // 4 for f in fft_sizes]
        self.win_lengths = win_lengths or fft_sizes
        self.window = window
        self.spectral_convergence_weight = spectral_convergence_weight
        self.log_magnitude_weight = log_magnitude_weight
        self.epsilon = epsilon

        # Register windows as buffers (nicht trainierbar)
        for i, (fft_size, win_length) in enumerate(zip(self.fft_sizes, self.win_lengths)):
            if window == "hann":
                win = torch.hann_window(win_length)
            elif window == "hamming":
                win = torch.hamming_window(win_length)
            elif window == "blackman":
                win = torch.blackman_window(win_length)
            else:
                win = torch.ones(win_length)

            self.register_buffer(f"window_{i}", win)

    def stft(
        self, audio: torch.Tensor, fft_size: int, hop_size: int, win_length: int, window: torch.Tensor
    ) -> torch.Tensor:
        """Compute STFT."""
        # torch.stft expects [batch, samples] or [samples], so squeeze channel dimension
        audio_2d = audio.squeeze(1)  # [batch, channels, samples] -> [batch, samples]
        return torch.stft(
            audio_2d,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
            normalized=False,
        )

    def spectral_convergence_loss(self, output_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        """Spectral convergence loss."""
        return torch.norm(target_mag - output_mag, p="fro") / (torch.norm(target_mag, p="fro") + self.epsilon)

    def log_magnitude_loss(self, output_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        """Log magnitude loss."""
        log_output = torch.log(output_mag + self.epsilon)
        log_target = torch.log(target_mag + self.epsilon)
        return F.l1_loss(log_output, log_target)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute multi-resolution STFT loss.

        Args:
            output: Predicted audio [batch, channels, time]
            target: Ground truth audio [batch, channels, time]

        Returns:
            loss: Combined multi-resolution loss
            details: Dictionary with per-resolution losses
        """
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        details = {}

        for i, (fft_size, hop_size, win_length) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f"window_{i}")

            # Compute STFT
            output_stft = self.stft(output, fft_size, hop_size, win_length, window)
            target_stft = self.stft(target, fft_size, hop_size, win_length, window)

            # Magnitude
            output_mag = torch.abs(output_stft)
            target_mag = torch.abs(target_stft)

            # Compute losses
            sc_loss = self.spectral_convergence_loss(output_mag, target_mag)
            mag_loss = self.log_magnitude_loss(output_mag, target_mag)

            total_sc_loss += sc_loss
            total_mag_loss += mag_loss

            details[f"sc_loss_{fft_size}"] = sc_loss.item()
            details[f"mag_loss_{fft_size}"] = mag_loss.item()

        # Average over all resolutions
        total_sc_loss /= len(self.fft_sizes)
        total_mag_loss /= len(self.fft_sizes)

        # Weighted combination
        total_loss = self.spectral_convergence_weight * total_sc_loss + self.log_magnitude_weight * total_mag_loss

        details["total_sc_loss"] = total_sc_loss.item()
        details["total_mag_loss"] = total_mag_loss.item()

        return total_loss, details


class PANNsPerceptualLoss(nn.Module):
    """
    PANNs-based High-Level Perceptual Loss.

    Nutzt Pre-trained PANNs (Pre-trained Audio Neural Networks) zur Feature-Extraktion
    und berechnet Distanz im Embedding-Space.

    Referenz: Kong et al. (2020): "PANNs: Large-Scale Pretrained Audio Neural Networks"
    """

    def __init__(
        self,
        panns_model_path: str | None = None,
        feature_layers: list[str] | None = None,
        feature_weights: list[float] | None = None,
        distance_metric: str = "l1",
    ) -> None:
        super().__init__()
        if feature_layers is None:
            feature_layers = ["conv_block1", "conv_block2", "conv_block3", "conv_block4"]

        try:
            # Lazy import für PANNs
            from plugins.panns_plugin import PANNSPlugin

            self.panns_available = True

            # Initialize PANNs model
            self.panns = PANNSPlugin()
            logger.info("PANNs model loaded for perceptual loss")

        except ImportError:
            logger.warning("PANNs not available, using fallback spectral features")
            self.panns_available = False

        self.feature_layers = feature_layers
        self.feature_weights = feature_weights or [1.0] * len(feature_layers)
        self.distance_metric = distance_metric

    def extract_features(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from audio using PANNs."""
        if not self.panns_available:
            # Fallback: Use spectral features
            return self._extract_spectral_features(audio)

        # PANNs expects [batch, samples]
        if audio.ndim == 3:
            audio = audio.squeeze(1)  # Remove channel dimension if present

        # Get PANNs embeddings (implementation depends on PANNs API)
        # This is a placeholder - actual implementation depends on PANNs integration
        features = {}

        # Extract multi-level features
        # Note: Actual feature extraction depends on PANNs model architecture
        features["high_level"] = audio.mean(dim=-1)  # Placeholder

        return features

    def _extract_spectral_features(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fallback: Extract spectral features."""
        # Compute mel-spectrogram as fallback
        n_fft = 2048
        hop_length = 512

        spec = torch.stft(
            audio.squeeze(1) if audio.ndim == 3 else audio,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            center=True,
        )

        mag = torch.abs(spec)

        # Mel filterbank (simplified)
        mel_features = mag.mean(dim=1)  # Simplified

        return {"spectral": mel_features}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute PANNs-based perceptual loss.

        Args:
            output: Predicted audio [batch, channels, time]
            target: Ground truth audio [batch, channels, time]

        Returns:
            loss: Perceptual loss in feature space
            details: Dictionary with feature-wise losses
        """
        # Extract features
        output_features = self.extract_features(output)
        target_features = self.extract_features(target)

        total_loss = 0.0
        details = {}

        # Compute distance in each feature space
        for key in output_features:
            output_feat = output_features[key]
            target_feat = target_features[key]

            if self.distance_metric == "l1":
                feat_loss = F.l1_loss(output_feat, target_feat)
            elif self.distance_metric == "l2":
                feat_loss = F.mse_loss(output_feat, target_feat)
            elif self.distance_metric == "cosine":
                feat_loss = 1.0 - F.cosine_similarity(output_feat.flatten(1), target_feat.flatten(1), dim=1).mean()
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")

            total_loss += feat_loss
            details[f"feat_loss_{key}"] = feat_loss.item()

        return total_loss, details


class PsychoacousticMaskingLoss(nn.Module):
    """
    Psychoacoustic Masking Loss basierend auf ITU-R BS.1387 (PEAQ).

    Berücksichtigt:
    - Frequenz-Masking (simultaneous masking)
    - Temporal Masking (pre- and post-masking)
    - Kritische Bänder (Bark scale)
    """

    def __init__(self, sr: int = 48000, n_fft: int = 2048, hop_length: int = 512, n_bark_bands: int = 24):
        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bark_bands = n_bark_bands

        # Bark scale boundaries (approximation)
        self.bark_boundaries = self._compute_bark_boundaries()
        self.register_buffer("_stft_window", torch.hann_window(self.n_fft), persistent=False)

    def _compute_bark_boundaries(self) -> torch.Tensor:
        """Compute Bark scale band boundaries."""
        # Bark scale formula: z = 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500)^2)
        # Simplified linear spacing for implementation
        max_freq = self.sr / 2
        freqs = torch.linspace(0, max_freq, self.n_bark_bands + 1)

        # Convert to FFT bin indices
        bins = (freqs / max_freq * (self.n_fft // 2)).long()

        return bins

    def compute_masking_threshold(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute psychoacoustic masking threshold.

        Simplified version - full PEAQ implementation would be more complex.
        """
        # Group into Bark bands
        bark_magnitudes = []

        for i in range(len(self.bark_boundaries) - 1):
            start_bin = self.bark_boundaries[i]
            end_bin = self.bark_boundaries[i + 1]

            band_mag = magnitude[:, start_bin:end_bin, :].mean(dim=1, keepdim=True)
            bark_magnitudes.append(band_mag)

        bark_mag = torch.cat(bark_magnitudes, dim=1)

        # Compute masking threshold (simplified spreading function)
        # In full PEAQ, this would involve complex spreading functions
        masking_threshold = bark_mag * 0.1  # Simplified: 10% of signal as threshold

        return masking_threshold

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute psychoacoustic masking loss.

        Args:
            output: Predicted audio [batch, channels, time]
            target: Ground truth audio [batch, channels, time]

        Returns:
            loss: Psychoacoustically weighted loss
            details: Dictionary with loss components
        """
        stft_window = self._stft_window.to(device=output.device, dtype=output.dtype)

        # Compute STFT
        output_stft = torch.stft(
            output.squeeze(1) if output.ndim == 3 else output,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=stft_window,
            return_complex=True,
            center=True,
        )

        target_stft = torch.stft(
            target.squeeze(1) if target.ndim == 3 else target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=stft_window,
            return_complex=True,
            center=True,
        )

        output_mag = torch.abs(output_stft)
        target_mag = torch.abs(target_stft)

        # Compute masking threshold from target
        masking_threshold = self.compute_masking_threshold(target_mag)

        # Compute error
        error = torch.abs(output_mag - target_mag)

        # Group error into Bark bands
        bark_errors = []
        for i in range(len(self.bark_boundaries) - 1):
            start_bin = self.bark_boundaries[i]
            end_bin = self.bark_boundaries[i + 1]

            band_error = error[:, start_bin:end_bin, :].mean(dim=1, keepdim=True)
            bark_errors.append(band_error)

        bark_error = torch.cat(bark_errors, dim=1)

        # Weight errors by masking threshold
        # Errors above threshold are weighted more heavily
        weighted_error = torch.where(
            bark_error > masking_threshold,
            bark_error * 2.0,  # Double weight for audible errors
            bark_error * 0.5,  # Half weight for masked errors
        )

        loss = weighted_error.mean()

        details = {"psychoacoustic_loss": loss.item(), "avg_masking_threshold": masking_threshold.mean().item()}

        return loss, details


class MusicalFeatureLoss(nn.Module):
    """
    Musical Feature Loss für Harmonic, Rhythmic und Timbral Eigenschaften.

    Basiert auf:
    - Harmonic-to-Noise Ratio (HNR)
    - Onset Detection Consistency
    - Spectral Centroid/Rolloff
    """

    def __init__(
        self, sr: int = 48000, harmonic_weight: float = 1.0, rhythmic_weight: float = 0.8, timbral_weight: float = 0.8
    ):
        super().__init__()

        self.sr = sr
        self.harmonic_weight = harmonic_weight
        self.rhythmic_weight = rhythmic_weight
        self.timbral_weight = timbral_weight
        self.register_buffer("_timbral_stft_window", torch.hann_window(2048), persistent=False)

    def compute_harmonic_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss based on harmonic content preservation."""
        # Simplified: Compare low-frequency energy ratios
        # Full implementation would use proper harmonic analysis

        # Low-pass filter for fundamental frequencies
        # Placeholder implementation
        output_harmonic = output  # Would apply harmonic separation
        target_harmonic = target

        return F.mse_loss(output_harmonic, target_harmonic)

    def compute_rhythmic_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss based on rhythmic consistency."""
        # Compute onset envelope similarity
        # Simplified implementation

        # Compute energy envelope
        output_envelope = output.abs().mean(dim=1) if output.ndim == 3 else output.abs()
        target_envelope = target.abs().mean(dim=1) if target.ndim == 3 else target.abs()

        # Compare envelopes
        return F.mse_loss(output_envelope, target_envelope)

    def compute_timbral_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss based on timbral characteristics."""
        # Compute spectral features
        # Simplified: Compare spectral centroids

        n_fft = 2048
        hop_length = 512
        stft_window = self._timbral_stft_window.to(device=output.device, dtype=output.dtype)

        output_spec = torch.stft(
            output.squeeze(1) if output.ndim == 3 else output,
            n_fft=n_fft,
            hop_length=hop_length,
            window=stft_window,
            return_complex=True,
            center=True,
        )

        target_spec = torch.stft(
            target.squeeze(1) if target.ndim == 3 else target,
            n_fft=n_fft,
            hop_length=hop_length,
            window=stft_window,
            return_complex=True,
            center=True,
        )

        output_mag = torch.abs(output_spec)
        target_mag = torch.abs(target_spec)

        # Spectral centroid (weighted frequency mean)
        freqs = torch.linspace(0, self.sr / 2, n_fft // 2 + 1, device=output.device)
        freqs = freqs.view(1, -1, 1)

        output_centroid = (output_mag * freqs).sum(dim=1) / (output_mag.sum(dim=1) + 1e-8)
        target_centroid = (target_mag * freqs).sum(dim=1) / (target_mag.sum(dim=1) + 1e-8)

        return F.mse_loss(output_centroid, target_centroid)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute musical feature loss.

        Args:
            output: Predicted audio [batch, channels, time]
            target: Ground truth audio [batch, channels, time]

        Returns:
            loss: Combined musical feature loss
            details: Dictionary with component losses
        """
        harmonic_loss = self.compute_harmonic_loss(output, target)
        rhythmic_loss = self.compute_rhythmic_loss(output, target)
        timbral_loss = self.compute_timbral_loss(output, target)

        total_loss = (
            self.harmonic_weight * harmonic_loss
            + self.rhythmic_weight * rhythmic_loss
            + self.timbral_weight * timbral_loss
        )

        details = {
            "harmonic_loss": harmonic_loss.item(),
            "rhythmic_loss": rhythmic_loss.item(),
            "timbral_loss": timbral_loss.item(),
        }

        return total_loss, details


class CombinedPerceptualLoss(nn.Module):
    """
    Combined Perceptual Loss combining all perceptual loss components.

    Optimale Gewichtung für musikalische Audio-Restauration.
    """

    def __init__(
        self,
        sr: int = 48000,
        stft_weight: float = 1.0,
        panns_weight: float = 0.8,
        psychoacoustic_weight: float = 0.6,
        musical_weight: float = 1.2,
        use_panns: bool = True,
        use_psychoacoustic: bool = True,
        use_musical: bool = True,
    ):
        super().__init__()

        self.stft_weight = stft_weight
        self.panns_weight = panns_weight
        self.psychoacoustic_weight = psychoacoustic_weight
        self.musical_weight = musical_weight

        # Initialize loss components
        self.stft_loss = MultiResolutionSTFTLoss()

        if use_panns:
            self.panns_loss = PANNsPerceptualLoss()
        else:
            self.panns_loss = None

        if use_psychoacoustic:
            self.psychoacoustic_loss = PsychoacousticMaskingLoss(sr=sr)
        else:
            self.psychoacoustic_loss = None

        if use_musical:
            self.musical_loss = MusicalFeatureLoss(sr=sr)
        else:
            self.musical_loss = None

        logger.info("CombinedPerceptualLoss initialized with sr=%s", sr)
        logger.info("  STFT weight: %s", stft_weight)
        logger.info("  PANNs weight: %s (enabled: %s)", panns_weight, use_panns)
        logger.info("  Psychoacoustic weight: %s (enabled: %s)", psychoacoustic_weight, use_psychoacoustic)
        logger.info("  Musical weight: %s (enabled: %s)", musical_weight, use_musical)

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, return_details: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined perceptual loss.

        Args:
            output: Predicted audio [batch, channels, time]
            target: Ground truth audio [batch, channels, time]
            return_details: If True, return detailed loss breakdown

        Returns:
            loss: Combined perceptual loss
            details: (optional) Dictionary with all component losses
        """
        total_loss = 0.0
        all_details = {}

        # 1. Multi-Resolution STFT Loss
        stft_loss, stft_details = self.stft_loss(output, target)
        total_loss += self.stft_weight * stft_loss
        all_details.update({f"stft_{k}": v for k, v in stft_details.items()})

        # 2. PANNs Perceptual Loss
        if self.panns_loss is not None:
            panns_loss, panns_details = self.panns_loss(output, target)
            total_loss += self.panns_weight * panns_loss
            all_details.update({f"panns_{k}": v for k, v in panns_details.items()})

        # 3. Psychoacoustic Masking Loss
        if self.psychoacoustic_loss is not None:
            psych_loss, psych_details = self.psychoacoustic_loss(output, target)
            total_loss += self.psychoacoustic_weight * psych_loss
            all_details.update({f"psych_{k}": v for k, v in psych_details.items()})

        # 4. Musical Feature Loss
        if self.musical_loss is not None:
            musical_loss, musical_details = self.musical_loss(output, target)
            total_loss += self.musical_weight * musical_loss
            all_details.update({f"musical_{k}": v for k, v in musical_details.items()})

        all_details["total_perceptual_loss"] = total_loss.item()

        if return_details:
            return total_loss, all_details
        return total_loss


# Example usage and testing
if __name__ == "__main__":
    # Test perceptual loss
    batch_size = 2
    channels = 1
    duration = 2  # seconds
    sr = 48000
    samples = duration * sr

    # Create dummy audio
    output = torch.randn(batch_size, channels, samples)
    target = torch.randn(batch_size, channels, samples)

    # Test Combined Perceptual Loss
    loss_fn = CombinedPerceptualLoss(sr=sr)
    loss, details = loss_fn(output, target, return_details=True)

    logger.debug("Total Perceptual Loss: %.4f", loss.item())
    logger.debug("\nDetailed Breakdown:")
    for key, value in details.items():
        logger.debug("  %s: %.4f", key, value)
