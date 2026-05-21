"""SOTA vocal model router for vocal-first music restoration (§SMR-1).

Centralizes model choice for pre-phase vocal/instrumental stem restoration:
BS-RoFormer → Demucs v4 → MDX23C for separation, and MIIPHER → SGMSE+ →
DeepFilterNet for vocal NR.  The router is deliberately adapter-only: it does
not invent new DSP, it selects the strongest available local plugin and returns
explicit fallback metadata.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from operator import methodcaller
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StemSeparationRouteResult:
    """Result of routed vocal/instrumental separation."""

    vocal: np.ndarray
    instrumental: np.ndarray
    success: bool
    model_used: str
    fallback_chain: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class EnhancementRouteResult:
    """Result of routed stem enhancement."""

    audio: np.ndarray
    success: bool
    model_used: str
    fallback_chain: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class SotaVocalModelRouter:
    """Routes vocal-centric separation and NR through the best local model."""

    _SEPARATION_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]

    def separate_vocal_instrumental(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        panns_singing: float = 0.0,
        ctx: dict[str, Any] | None = None,
    ) -> StemSeparationRouteResult:
        """Separate mix into vocal + full instrumental stems using routed SOTA plugins."""
        del ctx  # Reserved for later VRAM/material routing.
        reference = np.asarray(audio, dtype=np.float32)
        attempts: list[str] = []
        capability_report = self._capability_report()

        if panns_singing >= 0.35:
            try:
                from plugins.bs_roformer_plugin import get_bs_roformer  # pylint: disable=import-outside-toplevel

                separator = get_bs_roformer()
                result = separator.separate(reference, sr, stems=self._SEPARATION_STEMS)
                stems_obj = getattr(result, "stems", {})
                model_used = str(getattr(result, "model_used", "bs_roformer"))
                if isinstance(stems_obj, dict) and stems_obj and self._is_primary_roformer_model(model_used):
                    vocal, instrumental = self._stems_to_vocal_instr(stems_obj, reference)
                    return StemSeparationRouteResult(
                        vocal=vocal,
                        instrumental=instrumental,
                        success=True,
                        model_used=model_used,
                        fallback_chain=attempts.copy(),
                        metadata={
                            "confidence": float(getattr(result, "confidence", 0.0)),
                            "sdri_db": float(getattr(result, "sdri_db", 0.0)),
                            "roformer_model_loaded": bool(getattr(separator, "_model_loaded", False)),
                            "capability_status": self._capability_status(capability_report, "melbandroformer"),
                        },
                    )
                if isinstance(stems_obj, dict) and stems_obj:
                    attempts.append(f"bs_roformer:{model_used}")
                    logger.debug(
                        "§SMR-1 BS/MelBand-RoFormer returned fallback model=%s; trying next separator",
                        model_used,
                    )
                    raise RuntimeError("roformer_fallback_result")
                attempts.append("bs_roformer:empty_stems")
            except Exception as exc:  # pylint: disable=broad-except
                if str(exc) != "roformer_fallback_result":
                    attempts.append(f"bs_roformer:{type(exc).__name__}")
                logger.debug("§SMR-1 BS-RoFormer separation unavailable: %s", exc)

        try:
            from plugins.demucs_v4_plugin import get_demucs_plugin  # pylint: disable=import-outside-toplevel

            stems = get_demucs_plugin().separate(reference, sr)
            if isinstance(stems, dict) and stems:
                vocal, instrumental = self._stems_to_vocal_instr(stems, reference)
                return StemSeparationRouteResult(
                    vocal=vocal,
                    instrumental=instrumental,
                    success=True,
                    model_used="demucs_v4",
                    fallback_chain=attempts.copy(),
                    metadata={"capability_status": self._capability_status(capability_report, "demucs_v4")},
                )
            attempts.append("demucs_v4:empty_stems")
        except Exception as exc:  # pylint: disable=broad-except
            attempts.append(f"demucs_v4:{type(exc).__name__}")
            logger.debug("§SMR-1 Demucs separation unavailable: %s", exc)

        try:
            from plugins.mdx23c_plugin import get_mdx23c_plugin  # pylint: disable=import-outside-toplevel

            stems = get_mdx23c_plugin().separate_all_stems(reference, sr, stems=["vocals", "inst"])
            if isinstance(stems, dict) and stems:
                vocal = self._coerce_like(stems.get("vocals", np.zeros_like(reference)), reference)
                instrumental = self._coerce_like(stems.get("inst", reference - vocal), reference)
                return StemSeparationRouteResult(
                    vocal=vocal,
                    instrumental=instrumental,
                    success=True,
                    model_used="mdx23c",
                    fallback_chain=attempts.copy(),
                    metadata={"capability_status": "sota_fallback"},
                )
            attempts.append("mdx23c:empty_stems")
        except Exception as exc:  # pylint: disable=broad-except
            attempts.append(f"mdx23c:{type(exc).__name__}")
            logger.debug("§SMR-1 MDX23C separation unavailable: %s", exc)

        return StemSeparationRouteResult(
            vocal=np.zeros_like(reference, dtype=np.float32),
            instrumental=reference.copy(),
            success=False,
            model_used="dsp_fallback_required",
            fallback_chain=attempts,
        )

    def enhance_vocal(
        self,
        vocal_stem: np.ndarray,
        sr: int,
        *,
        energy_bias_db: float = -6.0,
        noise_snr_db: float = 0.0,
    ) -> EnhancementRouteResult:
        """Enhance vocal stem through MIIPHER → SGMSE+ → DFN with explicit fallback metadata."""
        reference = np.asarray(vocal_stem, dtype=np.float32)
        attempts: list[str] = []
        capability_report = self._capability_report()

        try:
            from plugins.miipher_plugin import get_miipher_plugin  # pylint: disable=import-outside-toplevel

            miipher = get_miipher_plugin()
            model_loaded = bool(getattr(miipher, "_model_loaded", False))
            is_productive = False
            try:
                is_productive = bool(methodcaller("is_productive")(miipher))
            except Exception as prod_exc:  # pylint: disable=broad-except
                logger.debug("§SMR-1 MIIPHER adapter productivity check failed: %s", prod_exc)
            if not model_loaded and not is_productive:
                attempts.append("miipher:not_loaded")
                raise RuntimeError("miipher_model_not_loaded")
            out = miipher.enhance(reference, sr, noise_snr_db=noise_snr_db)
            out_arr = self._coerce_like(out, reference)
            route_metadata = getattr(miipher, "route_metadata", {})
            if not isinstance(route_metadata, dict):
                route_metadata = {}
            route_status = str(
                route_metadata.get(
                    "capability_status",
                    self._capability_status(capability_report, "miipher"),
                )
            )
            route_model = str(route_metadata.get("model_used", "miipher" if model_loaded else "miipher_adapter"))
            if route_status == "dsp_fallback":
                attempts.append(f"miipher:{route_model}")
                raise RuntimeError("miipher_adapter_dsp_fallback")
            return EnhancementRouteResult(
                audio=out_arr,
                success=True,
                model_used=route_model,
                fallback_chain=attempts.copy(),
                metadata={
                    "miipher_model_loaded": model_loaded,
                    "miipher_adapter_productive": is_productive,
                    "energy_bias_db": float(energy_bias_db),
                    "capability_status": route_status,
                    "miipher_route_metadata": dict(route_metadata),
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            if str(exc) not in {"miipher_model_not_loaded", "miipher_adapter_dsp_fallback"}:
                attempts.append(f"miipher:{type(exc).__name__}")
            logger.debug("§SMR-1 MIIPHER unavailable: %s", exc)

        try:
            from plugins.sgmse_plugin import get_sgmse_plugin  # pylint: disable=import-outside-toplevel

            sgmse = get_sgmse_plugin()
            raw = sgmse.enhance(reference, sr)
            out = getattr(raw, "audio", raw)
            sgmse_audio = self._coerce_like(out, reference)
            _sgmse_model_used = str(getattr(raw, "model_used", "sgmse_plus"))
            compensated = self._compensate_missing_miipher(
                sgmse_audio,
                reference,
                sr,
                energy_bias_db=energy_bias_db,
                base_model_used=_sgmse_model_used,
            )
            metadata = {
                "energy_bias_db": float(energy_bias_db),
                "miipher_model_loaded": False,
                "sgmse_model_loaded": bool(getattr(sgmse, "_model_loaded", False)),
                "capability_status": self._capability_status(capability_report, "sgmse_plus"),
            }
            metadata.update(compensated.get("metadata", {}))
            return EnhancementRouteResult(
                audio=self._coerce_like(compensated.get("audio", sgmse_audio), reference),
                success=True,
                model_used=str(compensated.get("model_used", _sgmse_model_used)),
                fallback_chain=attempts.copy(),
                metadata=metadata,
            )
        except Exception as exc:  # pylint: disable=broad-except
            attempts.append(f"sgmse_plus:{type(exc).__name__}")
            logger.debug("§SMR-1 SGMSE+ unavailable: %s", exc)

        dfn_result = self.enhance_instrumental(reference, sr, energy_bias_db=energy_bias_db)
        dfn_result.fallback_chain = attempts + dfn_result.fallback_chain
        if dfn_result.success:
            dfn_result.model_used = f"vocal_{dfn_result.model_used}"
        return dfn_result

    def _compensate_missing_miipher(
        self,
        sgmse_audio: np.ndarray,
        reference: np.ndarray,
        sr: int,
        *,
        energy_bias_db: float,
        base_model_used: str,
    ) -> dict[str, object]:
        """Kompensiert fehlendes MIIPHER bestmöglich via DFN + optionalem HNR-Blend."""
        base_audio = self._coerce_like(sgmse_audio, reference)
        metadata: dict[str, object] = {
            "miipher_compensation_active": True,
            "miipher_compensation_dfn_applied": False,
            "miipher_compensation_hnr_applied": False,
        }
        model_used = str(base_model_used or "sgmse_plus")

        dfn_result = self.enhance_instrumental(base_audio, sr, energy_bias_db=energy_bias_db)
        if not dfn_result.success:
            metadata["miipher_compensation_dfn_reason"] = "deepfilternet_unavailable"
            return {"audio": base_audio, "model_used": model_used, "metadata": metadata}

        compensated_audio = self._coerce_like(dfn_result.audio, reference)
        metadata["miipher_compensation_dfn_applied"] = True
        metadata["miipher_compensation_dfn_model"] = str(dfn_result.model_used)
        model_used = f"sgmse_plus+{dfn_result.model_used}"

        try:
            from backend.core.dsp.hnr_guard import apply_hnr_blend  # pylint: disable=import-outside-toplevel

            compensated_audio = self._coerce_like(apply_hnr_blend(reference, compensated_audio, sr), reference)
            metadata["miipher_compensation_hnr_applied"] = True
            model_used = f"{model_used}+hnr_blend"
        except Exception as exc:  # pylint: disable=broad-except
            metadata["miipher_compensation_hnr_reason"] = type(exc).__name__

        return {"audio": compensated_audio, "model_used": model_used, "metadata": metadata}

    def enhance_instrumental(
        self,
        stem: np.ndarray,
        sr: int,
        *,
        energy_bias_db: float = -9.0,
    ) -> EnhancementRouteResult:
        """Enhance instrumental stem via DeepFilterNet v3.II."""
        reference = np.asarray(stem, dtype=np.float32)
        attempts: list[str] = []
        capability_report = self._capability_report()
        try:
            from plugins.deepfilternet_v3_ii_plugin import (  # pylint: disable=import-outside-toplevel
                get_deepfilternet_plugin,
            )

            out = get_deepfilternet_plugin().enhance(reference, sr, energy_bias_db=energy_bias_db)
            return EnhancementRouteResult(
                audio=self._coerce_like(out, reference),
                success=True,
                model_used="deepfilternet_v3_ii",
                fallback_chain=attempts,
                metadata={
                    "energy_bias_db": float(energy_bias_db),
                    "capability_status": self._capability_status(capability_report, "deepfilternet_v3_ii"),
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            attempts.append(f"deepfilternet_v3_ii:{type(exc).__name__}")
            logger.debug("§SMR-1 DeepFilterNet unavailable: %s", exc)
            return EnhancementRouteResult(
                audio=reference.copy(),
                success=False,
                model_used="none",
                fallback_chain=attempts,
                metadata={"energy_bias_db": float(energy_bias_db), "capability_status": "dsp_fallback"},
            )

    @staticmethod
    def _capability_report() -> dict[str, object]:
        try:
            from backend.core.dsp.model_capability_gate import (  # pylint: disable=import-outside-toplevel
                get_model_capability_gate,
            )

            return get_model_capability_gate().build_report()
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§SMR-1 model capability report unavailable: %s", exc)
            return {}

    @staticmethod
    def _capability_status(report: dict[str, object], name: str) -> str:
        try:
            capabilities = report.get("capabilities", {}) if isinstance(report, dict) else {}
            if isinstance(capabilities, dict):
                cap = capabilities.get(name, {})
                if isinstance(cap, dict):
                    return str(cap.get("status", "unknown"))
        except Exception:  # pylint: disable=broad-except
            pass
        return "unknown"

    @classmethod
    def _is_primary_roformer_model(cls, model_used: str) -> bool:
        """True only for actual BS/MelBand-RoFormer inference, not plugin fallbacks."""
        normalized = model_used.lower()
        return normalized in {"bs_roformer", "melbandroformer", "melband_roformer"}

    @classmethod
    def _stems_to_vocal_instr(
        cls,
        stems: dict[str, object],
        reference: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        vocal = cls._coerce_like(stems.get("vocals", np.zeros_like(reference)), reference)
        instr_parts = [
            cls._coerce_like(stem_audio, reference) for stem_name, stem_audio in stems.items() if stem_name != "vocals"
        ]
        if instr_parts:
            instrumental = np.sum(np.stack(instr_parts, axis=0), axis=0).astype(np.float32)
        else:
            instrumental = (np.asarray(reference, dtype=np.float32) - vocal).astype(np.float32)
        instrumental = np.clip(np.nan_to_num(instrumental, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        return vocal.astype(np.float32), instrumental.astype(np.float32)

    @staticmethod
    def _coerce_like(candidate: object, reference: np.ndarray) -> np.ndarray:
        """Gibt candidate as finite float32 audio with reference shape/layout zurück."""
        ref = np.asarray(reference, dtype=np.float32)
        try:
            arr = np.asarray(candidate, dtype=np.float32)
        except Exception:  # pylint: disable=broad-except
            return np.zeros_like(ref, dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2 and ref.ndim == 2 and arr.T.shape == ref.shape:
            arr = arr.T
        if ref.ndim == 1 and arr.ndim == 2:
            arr = arr.mean(axis=1 if arr.shape[0] == ref.shape[0] else 0)
        elif ref.ndim == 2 and arr.ndim == 1:
            arr = np.repeat(arr[:, None], ref.shape[1], axis=1)

        if arr.ndim != ref.ndim:
            return np.zeros_like(ref, dtype=np.float32)
        if arr.shape[0] < ref.shape[0]:
            pad_shape = list(arr.shape)
            pad_shape[0] = ref.shape[0] - arr.shape[0]
            arr = np.concatenate([arr, np.zeros(pad_shape, dtype=np.float32)], axis=0)
        elif arr.shape[0] > ref.shape[0]:
            arr = arr[: ref.shape[0]]

        if arr.shape != ref.shape:
            return np.zeros_like(ref, dtype=np.float32)
        return np.clip(arr.astype(np.float32), -1.0, 1.0)


_instance: SotaVocalModelRouter | None = None
_lock = threading.Lock()


def get_sota_vocal_model_router() -> SotaVocalModelRouter:
    """Thread-safe singleton accessor for :class:`SotaVocalModelRouter`."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SotaVocalModelRouter()
    return _instance


__all__ = [
    "EnhancementRouteResult",
    "SotaVocalModelRouter",
    "StemSeparationRouteResult",
    "get_sota_vocal_model_router",
]
