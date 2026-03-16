"""
sota_vocal_separator.py - SOTA-Source-Separation für Aurik 6.0
Produktive Integration von Hybrid Demucs/Banquet für Vocal-/Instrumenten-Separation.
"""

import logging
import os

import numpy as np
import onnxruntime as ort
logger = logging.getLogger(__name__)

MODEL_PATH_BANQUET = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/banquet/banquet_vinyl_final.onnx")
)
MODEL_PATH_UVR_MDX_NET = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/uvr_mdx_net/uvr_mdx_net_inst_hq_1.onnx")
)
MODEL_PATH_DEMUCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/demucs/htdemucs_6s.onnx"))


class SotaVocalSeparator:
    def __init__(self, model_path=None, use_uvr=False, use_demucs=False):
        if use_demucs:
            self.model_path = model_path or MODEL_PATH_DEMUCS
        elif use_uvr:
            self.model_path = model_path or MODEL_PATH_UVR_MDX_NET
        else:
            self.model_path = model_path or MODEL_PATH_BANQUET
        self.session = None
        if os.path.exists(self.model_path):
            self.session = ort.InferenceSession(self.model_path)
        else:
            logger.warning(f"[WARN] ONNX-Modell nicht gefunden: {self.model_path}")

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.session is None:
            logger.warning("[WARN] Kein ONNX-Modell geladen, Rückgabe des Originalsignals.")
            return audio
        x = audio.astype(np.float32)
        if x.ndim == 1:
            x = x[None, :]
        ort_inputs = {self.session.get.inputs()[0].name: x}
        try:
            ort_outs = self.session.run(None, ort_inputs)
            return np.asarray(ort_outs[0].squeeze())
        except Exception as e:
            logger.error(f"[ERROR] Inferenz fehlgeschlagen: {e}")
            return audio
