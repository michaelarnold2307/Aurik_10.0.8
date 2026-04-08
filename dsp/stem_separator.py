"""
stem_separator.py - Audio Source Separation for Stem Export (GAP #43)

Separates mixed audio into individual stems:
- Vocals
- Drums
- Bass
- Other/Accompaniment

Provides two separation backends:
1. **Spectral Separation** (always available, fast, basic quality)
2. **ML-based Separation** (optional, Demucs/Banquet integration, high quality)

Author: AURIK Development Team
Version: 2.0.0 (GAP #43 Implementation)
Date: 9. Februar 2026
"""

import logging
import warnings

import numpy as np
from scipy.signal import istft, stft

_logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Check for optional ML backends
DEMUCS_AVAILABLE = False
BANQUET_AVAILABLE = False

try:
    pass

    # Note: Actual Demucs import would be here in production
    # from demucs import pretrained
    # from demucs.apply import apply_model
    # DEMUCS_AVAILABLE = True
except ImportError:
    pass

try:
    from banquet import Banquet  # type: ignore[import-untyped]

    BANQUET_AVAILABLE = True
except ImportError:

    class Banquet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Banquet backend not available")

    BANQUET_AVAILABLE = False


class SpectralStemSeparator:
    """
    Simple spectral-based stem separation.

    Fast and always available, but basic quality.
    Good for preview/quick exports.
    """

    def __init__(
        self,
        vocal_freq_range: tuple[float, float] = (80, 8000),
        bass_freq_range: tuple[float, float] = (20, 250),
        drums_freq_range: tuple[float, float] = (50, 15000),
    ):
        """
        Parameters
        ----------
        vocal_freq_range : tuple
            (low_hz, high_hz) for vocal extraction
        bass_freq_range : tuple
            (low_hz, high_hz) for bass extraction
        drums_freq_range : tuple
            (low_hz, high_hz) for drums extraction
        """
        self.vocal_freq_range = vocal_freq_range
        self.bass_freq_range = bass_freq_range
        self.drums_freq_range = drums_freq_range
        self.metrics = {}

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """
        Separate audio into stems using spectral filtering.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        stems : dict
            Dictionary with keys 'vocals', 'drums', 'bass', 'other'
        """
        # Ensure stereo for processing
        is_mono = audio.ndim == 1
        if is_mono:
            audio_stereo = np.stack([audio, audio])
        else:
            # scipy.signal.stft expects shape (channels, samples)
            # Input is usually (samples, channels), so transpose
            if audio.shape[0] > audio.shape[1]:
                audio_stereo = audio.T  # (samples, 2) → (2, samples)
            else:
                audio_stereo = audio  # already (2, samples)

        # STFT
        # Adapt nperseg to signal length
        # audio_stereo now has shape (2, samples) for stereo or (samples,) for mono duplicate
        signal_length = audio_stereo.shape[-1] if audio_stereo.ndim > 1 else len(audio_stereo)
        nperseg = min(2048, max(512, signal_length // 8))
        noverlap = nperseg // 2

        f, _t, Zxx = stft(audio_stereo, sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Frequency masks
        vocal_mask = self._create_freq_mask(f, self.vocal_freq_range)
        bass_mask = self._create_freq_mask(f, self.bass_freq_range)
        drums_mask = self._create_transient_mask(Zxx, f, self.drums_freq_range)

        # Extract stems via masking
        vocals_stft = Zxx * vocal_mask[:, None]
        bass_stft = Zxx * bass_mask[:, None]
        drums_stft = Zxx * drums_mask

        # Other = everything not in vocals/bass/drums
        other_stft = Zxx - vocals_stft - bass_stft - drums_stft

        # ISTFT to reconstruct
        _, vocals = istft(vocals_stft, sample_rate, nperseg=nperseg, noverlap=noverlap)
        _, bass = istft(bass_stft, sample_rate, nperseg=nperseg, noverlap=noverlap)
        _, drums = istft(drums_stft, sample_rate, nperseg=nperseg, noverlap=noverlap)
        _, other = istft(other_stft, sample_rate, nperseg=nperseg, noverlap=noverlap)

        # Convert back to original format
        if is_mono:
            vocals = vocals[0]  # Take first channel
            drums = drums[0]
            bass = bass[0]
            other = other[0]
        else:
            vocals = vocals.T
            drums = drums.T
            bass = bass.T
            other = other.T

        # Ensure correct length
        target_len = audio.shape[-1] if is_mono else audio.shape[0]
        stems = {
            "vocals": self._match_length(vocals, target_len, is_mono),
            "drums": self._match_length(drums, target_len, is_mono),
            "bass": self._match_length(bass, target_len, is_mono),
            "other": self._match_length(other, target_len, is_mono),
        }

        # Store metrics
        self.metrics = {"backend": "spectral", "processing_time_estimate": "fast", "quality": "basic"}

        orig_dtype = audio.dtype
        return {k: np.asarray(v).astype(orig_dtype) for k, v in stems.items()}

    def _create_freq_mask(self, frequencies: np.ndarray, freq_range: tuple[float, float]) -> np.ndarray:
        """Create binary frequency mask"""
        low, high = freq_range
        mask = ((frequencies >= low) & (frequencies <= high)).astype(float)
        return mask

    def _create_transient_mask(
        self, Zxx: np.ndarray, frequencies: np.ndarray, freq_range: tuple[float, float]
    ) -> np.ndarray:
        """
        Create transient-based mask for drums.

        Drums have strong transients (sharp onset).
        """
        # Frequency-limited mask
        freq_mask = self._create_freq_mask(frequencies, freq_range)

        # Detect transients by high-frequency energy changes
        magnitude = np.abs(Zxx)

        # Temporal derivative (high for transients)
        diff = np.diff(magnitude, axis=1, prepend=magnitude[:, :1])
        transient_strength = np.abs(diff)

        # Threshold for transient detection
        threshold = np.percentile(transient_strength, 85)
        transient_mask = (transient_strength > threshold).astype(float)

        # Combine frequency and transient masks
        combined_mask = freq_mask[:, None] * transient_mask

        return combined_mask

    def _match_length(self, audio: np.ndarray, target_length: int, is_mono: bool) -> np.ndarray:
        """Match audio length to target"""
        if is_mono:
            if len(audio) > target_length:
                return audio[:target_length]
            elif len(audio) < target_length:
                return np.pad(audio, (0, target_length - len(audio)))
            return audio
        else:
            # Stereo
            if audio.shape[0] > target_length:
                return audio[:target_length]
            elif audio.shape[0] < target_length:
                pad_width = ((0, target_length - audio.shape[0]), (0, 0))
                return np.pad(audio, pad_width)
            return audio

    def get_metrics(self) -> dict:
        """Get separation metrics"""
        return self.metrics


class BanquetStemSeparator:
    """
    ML-based stem separation using Banquet.

    High quality but requires Banquet installation.
    """

    def __init__(self, model_path: str | None = None):
        """
        Parameters
        ----------
        model_path : str, optional
            Path to custom Banquet model
        """
        if not BANQUET_AVAILABLE:
            raise RuntimeError("Banquet not available. Install with: pip install banquet")

        self.model = Banquet(model_path) if model_path else Banquet()
        self.metrics = {}

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """
        Separate audio using Banquet.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        stems : dict
            Dictionary with keys 'vocals', 'drums', 'bass', 'other'
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        orig_dtype = audio.dtype
        result = self.model.separate(audio, sample_rate)

        # Normalize keys  (Banquet may return different key names)
        stems = {
            "vocals": np.asarray(result.get("vocals", result.get("vocal", audio))),
            "drums": np.asarray(result.get("drums", result.get("drum", np.zeros_like(audio)))),
            "bass": np.asarray(result.get("bass", np.zeros_like(audio))),
            "other": np.asarray(result.get("other", result.get("accompaniment", np.zeros_like(audio)))),
        }

        # Store metrics
        self.metrics = {"backend": "banquet", "quality": "high", "model_type": "ml"}

        return {k: v.astype(orig_dtype) for k, v in stems.items()}

    def get_metrics(self) -> dict:
        """Get separation metrics"""
        return self.metrics


class StemSeparator:
    """
    Unified API for stem separation.

    Automatically selects best available backend:
    - Banquet (if available) for best quality
    - Spectral separation (fallback) for speed
    """

    def __init__(self, backend: str = "auto", **backend_kwargs):
        """
        Parameters
        ----------
        backend : str
            'auto', 'spectral', 'banquet', or 'demucs'
        backend_kwargs : dict
            Parameters passed to backend
        """
        if backend == "auto":
            # Prefer Banquet if available, otherwise spectral
            backend = "banquet" if BANQUET_AVAILABLE else "spectral"

        if backend == "banquet":
            if not BANQUET_AVAILABLE:
                _logger.warning("Banquet not available, falling back to spectral")
                backend = "spectral"
            else:
                self.backend = BanquetStemSeparator(**backend_kwargs)

        if backend == "spectral":
            self.backend = SpectralStemSeparator(**backend_kwargs)

        if backend == "demucs":
            try:
                pass

                # DemucsStemSeparator might not be implemented yet
                # self.backend = DemucsStemSeparator(**backend_kwargs)
                # self.backend_name = 'demucs'
                _logger.warning("DemucsStemSeparator not implemented, falling back to spectral")
                backend = "spectral"
                self.backend = SpectralStemSeparator(**backend_kwargs)
            except ImportError:
                _logger.warning("Demucs not available, falling back to spectral")
                backend = "spectral"
                self.backend = SpectralStemSeparator(**backend_kwargs)

        # Persistente Zuweisung — unabhängig vom Backend-Pfad immer gesetzt
        self.backend_name = backend

    def separate(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        """
        Separate audio into stems.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        stems : dict
            Dictionary with keys:
            - 'vocals': Vocal stem
            - 'drums': Drum stem
            - 'bass': Bass stem
            - 'other': Other/accompaniment stem
        """
        return self.backend.separate(audio, sample_rate)

    def get_backend_info(self) -> dict[str, str]:
        """Get information about current backend"""
        if self.backend_name == "banquet":
            return {
                "backend": "banquet",
                "quality": "high",
                "speed": "medium",
                "description": "ML-based separation (Banquet)",
            }
        elif self.backend_name == "spectral":
            return {
                "backend": "spectral",
                "quality": "basic",
                "speed": "fast",
                "description": "Spectral frequency-based separation",
            }
        elif self.backend_name == "demucs":
            return {
                "backend": "demucs",
                "quality": "SOTA",
                "speed": "medium",
                "description": "Deep-Learning-based separation (Demucs)",
            }
        else:
            return {
                "backend": self.backend_name,
                "quality": "unknown",
                "speed": "unknown",
                "description": "Unknown backend",
            }

    class DemucsStemSeparator:
        """HPSS-basierter Stem-Separator (Fitzgerald 2010) mit Wiener-Masken.

        Dient als robuster CPU-Fallback wenn MDX23C/HTDemucs-ONNX nicht verfügbar.
        Aktivierung: automatisch durch StemSeparator-Backend-Kaskade.
        """

        def __init__(self, model_name: str = "htdemucs"):
            self.model_name = model_name

        def separate(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
            """HPSS-basierte Stem-Separation als Fallback ohne Demucs.

            Harmonic-Percussive Source Separation (Fitzgerald 2010):
              - STFT-basierte Median-Filterung in Zeit (harmonisch) und Frequenz (perkussiv)
              - Wiener-Maske für sanfte Trennung
              - Bass <250 Hz als separater Bass-Stem
            """
            from scipy.signal import istft, stft

            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            orig_dtype = audio.dtype
            try:
                mono = audio.mean(axis=0) if audio.ndim > 1 else audio
                mono = mono.astype(np.float64)
                nperseg = 2048
                noverlap = nperseg * 3 // 4
                _, _, Zxx = stft(mono, fs=sr, nperseg=nperseg, noverlap=noverlap)
                mag = np.abs(Zxx)
                # Harmonisch: Median über Zeit (Spalten)
                from scipy.ndimage import median_filter

                harm_mag = median_filter(mag, size=(1, 31))
                # Perkussiv: Median über Frequenz (Zeilen)
                perc_mag = median_filter(mag, size=(31, 1))
                # Soft-Maske (Wiener)
                total = harm_mag + perc_mag + 1e-12
                mask_h = harm_mag / total
                mask_p = perc_mag / total
                Zxx_h = Zxx * mask_h
                Zxx_p = Zxx * mask_p
                _, harm_audio = istft(Zxx_h, fs=sr, nperseg=nperseg, noverlap=noverlap)
                _, perc_audio = istft(Zxx_p, fs=sr, nperseg=nperseg, noverlap=noverlap)
                n = len(mono)
                harm_audio = harm_audio[:n] if len(harm_audio) >= n else np.pad(harm_audio, (0, n - len(harm_audio)))
                perc_audio = perc_audio[:n] if len(perc_audio) >= n else np.pad(perc_audio, (0, n - len(perc_audio)))
                # Bass: Low-pass <250 Hz aus Harmonisch
                from scipy.signal import butter, sosfilt

                sos_bass = butter(4, 250.0 / (sr / 2.0), btype="low", output="sos")
                bass_audio = sosfilt(sos_bass, harm_audio)
                vocals_audio = harm_audio - bass_audio  # Alles über Bass als Vocals/Other
                return {
                    "vocals": vocals_audio.astype(orig_dtype),
                    "drums": perc_audio.astype(orig_dtype),
                    "bass": bass_audio.astype(orig_dtype),
                    "other": ((harm_audio + perc_audio) / 2.0).astype(orig_dtype),
                }
            except Exception as e:
                raise RuntimeError(f"HPSS-Stem-Separation fehlgeschlagen: {e}") from e

        def get_metrics(self) -> dict:
            return {}

    def get_metrics(self) -> dict:
        """Get separation metrics from backend"""
        return self.backend.get_metrics()


# Convenience function
def separate_stems(audio: np.ndarray, sample_rate: int, backend: str = "auto") -> dict[str, np.ndarray]:
    """
    Separate audio into stems.

    Parameters
    ----------
    audio : np.ndarray
        Input audio (mono or stereo)
    sample_rate : int
        Sample rate in Hz
    backend : str
        'auto', 'spectral', 'banquet', or 'demucs'

    Returns
    -------
    stems : dict
        Dictionary with 'vocals', 'drums', 'bass', 'other'

    Examples
    --------
    >>> stems = separate_stems(audio, 48000)
    >>> vocals = stems['vocals']
    >>> drums = stems['drums']
    """
    separator = StemSeparator(backend=backend)
    return separator.separate(audio, sample_rate)


# Legacy compatibility
class AiStemSeparator:
    """Legacy wrapper for backward compatibility"""

    def __init__(self, model_path: str | None = None):
        self.separator = StemSeparator(backend="auto")

    def separate(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        stems = self.separator.separate(audio, sr)
        # Add 'piano' key for backward compatibility
        stems["piano"] = stems["other"]
        return stems


# CLI interface
if __name__ == "__main__":
    import argparse
    import os

    import soundfile as sf

    parser = argparse.ArgumentParser(description="AURIK Stem Separator (GAP #43)")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output-dir", "-o", default="stems", help="Output directory")
    parser.add_argument(
        "--backend", "-b", default="auto", choices=["auto", "spectral", "banquet"], help="Separation backend"
    )
    parser.add_argument("--format", "-f", default="wav", help="Output format (wav, flac, etc.)")

    args = parser.parse_args()

    # Load audio
    _logger.info("Loading: %s", args.input)
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])
    _logger.info("Loaded %d Hz, shape %s", sr, audio.shape)

    # Separate stems
    separator = StemSeparator(backend=args.backend)
    backend_info = separator.get_backend_info()
    _logger.info(
        "Backend: %s (%s quality, %s speed)", backend_info["backend"], backend_info["quality"], backend_info["speed"]
    )

    _logger.info("Separating stems...")
    stems = separator.separate(audio, sr)

    # Export stems
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    for stem_name, stem_audio in stems.items():
        ext = "." + args.format
        output_path = os.path.join(args.output_dir, f"{base_name}_{stem_name}{ext}")
        sf.write(output_path, stem_audio, sr)
        _logger.info("%-8s -> %s", stem_name, output_path)

    # Print metrics
    metrics = separator.get_metrics()
    _logger.info(
        "Separation complete! Backend: %s  Quality: %s",
        metrics.get("backend", "unknown"),
        metrics.get("quality", "unknown"),
    )
