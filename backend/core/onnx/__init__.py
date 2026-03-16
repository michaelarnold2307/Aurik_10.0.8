"""
ONNX Runtime Infrastructure for AURIK v8

This module provides ONNX Runtime integration for 3-6× speedup of ML model inference.

Key Components:
- OptimizedONNXModel: ONNX Runtime wrapper for model inference
- ONNXConverter: PyTorch → ONNX converter
- ModelQuantizer: FP32 → INT8 quantization
- FallbackManager: Automatic ONNX → PyTorch fallback
- ONNXModelWithFallback: Combined ONNX + PyTorch with auto fallback

Expected Speedup:
- ONNX: 1.5-2× faster than PyTorch
- Quantization: 2-3× additional speedup
- Total: 3-6× faster inference

Usage:
    from backend.core.onnx import OptimizedONNXModel, ModelQuantizer

    # Load ONNX model
    model = OptimizedONNXModel("model.onnx", model_type='denoising')

    # Process audio
    output = model.process(audio)

    # Quantize model
    quantizer = ModelQuantizer()
    quantizer.quantize("model.onnx", "model_quantized.onnx")
"""

from .converter import ConversionConfig, ModelSpecificConverter, ONNXConverter
from .fallback import FallbackEvent, FallbackManager, FallbackReason, FallbackStats, ONNXModelWithFallback
from .model_info import ModelInfo, ONNXModelStatus
from .plugin_manager import ONNXPluginManager, load_model, process_audio
from .quantizer import ModelQuantizer, QuantizationConfig, QuantizationType
from .runtime import ONNXInferenceSession, ONNXProvider, OptimizedONNXModel

__all__ = [
    # Runtime
    "OptimizedONNXModel",
    "ONNXInferenceSession",
    "ONNXProvider",
    # Model Info
    "ModelInfo",
    "ONNXModelStatus",
    # Converter
    "ONNXConverter",
    "ConversionConfig",
    "ModelSpecificConverter",
    # Quantizer
    "ModelQuantizer",
    "QuantizationConfig",
    "QuantizationType",
    # Fallback
    "FallbackManager",
    "ONNXModelWithFallback",
    "FallbackReason",
    "FallbackEvent",
    "FallbackStats",
    # Plugin Manager
    "ONNXPluginManager",
    "load_model",
    "process_audio",
]

__version__ = "1.0.0"
