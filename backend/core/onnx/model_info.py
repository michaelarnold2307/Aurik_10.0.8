"""
Model information and metadata for ONNX models.

Defines model registry structure and model status tracking.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass
class ModelInfo:
    """
    ONNX model metadata for audio processing.

    Attributes:
        model_path: Path to ONNX model file
        input_name: Input tensor name
        output_name: Output tensor name
        input_shape: Expected input shape (batch, samples)
        sample_rate: Audio sample rate (Hz)
        model_type: Model category (denoising, separation, enhancement)
        quantized: Whether model is INT8 quantized
        opset_version: ONNX opset version used
    """

    model_path: Path
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    sample_rate: int
    model_type: str
    quantized: bool = False
    opset_version: int = 14

    def __post_init__(self):
        """Validate model metadata."""
        if not isinstance(self.model_path, Path):
            self.model_path = Path(self.model_path)

        if self.model_type not in ["denoising", "separation", "enhancement", "analysis"]:
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                "Must be one of: denoising, separation, enhancement, analysis"
            )

        if self.sample_rate not in [16000, 22050, 32000, 44100, 48000, 96000]:
            import warnings

            warnings.warn(f"Unusual sample rate: {self.sample_rate} Hz. " f"Typical rates: 16k, 44.1k, 48k")


class ONNXModelStatus(Enum):
    """Model availability and readiness status."""

    READY = "ready"
    LOADING = "loading"
    ERROR = "error"
    NOT_FOUND = "not_found"
    FALLBACK_ACTIVE = "fallback_active"
