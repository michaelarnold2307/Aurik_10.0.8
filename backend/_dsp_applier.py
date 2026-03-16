"""Shim — forwards to canonical backend.core.regulator._dsp_applier."""

from backend.core.regulator._dsp_applier import (  # noqa: F401
    _ALWAYS_APPLY,
    _SKIP_GATE_THRESHOLD_DB,
    _apply_dsp_module,
    _compute_snr_db,
    apply_dsp_chain,
    apply_dsp_chain_tuple,
    compressor,
    dsp_effects,
    enhancer,
    eq,
    limiter,
)

__all__ = [
    "apply_dsp_chain",
    "apply_dsp_chain_tuple",
    "_apply_dsp_module",
    "_compute_snr_db",
    "_SKIP_GATE_THRESHOLD_DB",
    "_ALWAYS_APPLY",
    "dsp_effects",
    "eq",
    "compressor",
    "limiter",
    "enhancer",
]
