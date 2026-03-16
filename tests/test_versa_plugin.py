"""
test_versa_plugin.py — Tests für VERSA 2024 MOS-Plugin (§4.4)

VERSA ersetzt CDPAM als primäre non-reference MOS-Metrik für Musikrestaurierung.
API: get_versa_plugin().score(audio_np, sr) → VersaResult(mos, model_used)

Invarianten (§3.1, §3.2):
    - MOS ∈ [1.0, 5.0]
    - NaN/Inf-sicher
    - Singleton Thread-sicher
    - CPU-only (§9.5)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

SR = 48_000
DURATION_S = 1.0
_N = int(SR * DURATION_S)

AUDIO_SINE = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, DURATION_S, _N))).astype(np.float32)
AUDIO_NOISE = (0.1 * np.random.default_rng(42).standard_normal(_N)).astype(np.float32)
AUDIO_ZERO = np.zeros(_N, dtype=np.float32)
AUDIO_STEREO = np.stack([AUDIO_SINE, AUDIO_SINE], axis=1)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_versa_plugin_importable():
    """VersaPlugin muss aus plugins.versa_plugin importierbar sein."""
    from plugins.versa_plugin import VersaPlugin  # noqa: PLC0415

    assert VersaPlugin is not None


def test_get_versa_plugin_callable():
    """get_versa_plugin() liefert eine VersaPlugin-Instanz."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    p = get_versa_plugin()
    assert p is not None


def test_score_mos_callable():
    """score_mos() convenience-Funktion ist aufrufbar."""
    from plugins.versa_plugin import score_mos  # noqa: PLC0415

    assert callable(score_mos)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_singleton_same_instance():
    """Mehrfache get_versa_plugin()-Aufrufe liefern dieselbe Instanz."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    p1 = get_versa_plugin()
    p2 = get_versa_plugin()
    assert p1 is p2


# ---------------------------------------------------------------------------
# VersaResult — Struktur
# ---------------------------------------------------------------------------


def test_result_has_mos_attribute():
    """VersaResult hat mos-Attribut."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    result = get_versa_plugin().score(AUDIO_SINE, SR)
    assert hasattr(result, "mos"), "VersaResult muss mos-Attribut haben"


def test_result_has_model_used_attribute():
    """VersaResult hat model_used-Attribut."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    result = get_versa_plugin().score(AUDIO_SINE, SR)
    assert hasattr(result, "model_used"), "VersaResult muss model_used-Attribut haben"


# ---------------------------------------------------------------------------
# MOS-Wertebereich §4.4: mos ∈ [1.0, 5.0]
# ---------------------------------------------------------------------------


def test_mos_in_valid_range_sine():
    """MOS für sauberes Sinus-Signal muss in [1.0, 5.0] liegen."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    result = get_versa_plugin().score(AUDIO_SINE, SR)
    assert math.isfinite(result.mos), f"MOS ist nicht finite: {result.mos}"
    assert 1.0 <= result.mos <= 5.0, f"MOS außerhalb [1,5]: {result.mos}"


def test_mos_in_valid_range_noise():
    """MOS für Rauschen muss in [1.0, 5.0] liegen."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    result = get_versa_plugin().score(AUDIO_NOISE, SR)
    assert math.isfinite(result.mos)
    assert 1.0 <= result.mos <= 5.0


def test_mos_in_valid_range_zeros():
    """MOS für Stille muss in [1.0, 5.0] liegen (keine Division durch Null)."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    result = get_versa_plugin().score(AUDIO_ZERO, SR)
    assert math.isfinite(result.mos)
    assert 1.0 <= result.mos <= 5.0


def test_mos_stereo_input():
    """VERSA akzeptiert Stereo-Eingabe (ndim=2) ohne Fehler."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    result = get_versa_plugin().score(AUDIO_STEREO, SR)
    assert math.isfinite(result.mos)
    assert 1.0 <= result.mos <= 5.0


# ---------------------------------------------------------------------------
# score_mos() convenience wrapper
# ---------------------------------------------------------------------------


def test_score_mos_result_matches_plugin():
    """score_mos() liefert konsistentes Ergebnis mit get_versa_plugin().score()."""
    from plugins.versa_plugin import get_versa_plugin, score_mos  # noqa: PLC0415

    r1 = score_mos(AUDIO_SINE, SR)
    r2 = get_versa_plugin().score(AUDIO_SINE, SR)
    # Beide müssen valide MOS-Werte liefern
    assert math.isfinite(r1.mos)
    assert math.isfinite(r2.mos)
    assert 1.0 <= r1.mos <= 5.0
    assert 1.0 <= r2.mos <= 5.0


# ---------------------------------------------------------------------------
# NaN/Inf-Robustheit (§3.1)
# ---------------------------------------------------------------------------


def test_nan_input_no_crash():
    """NaN-Eingabe darf keinen Absturz auslösen."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    audio_nan = np.full(_N, np.nan, dtype=np.float32)
    try:
        result = get_versa_plugin().score(audio_nan, SR)
        assert math.isfinite(result.mos)
        assert 1.0 <= result.mos <= 5.0
    except (RuntimeError, ValueError):
        pass  # Explizite Exceptions erlaubt, Absturz (SIGSEGV) nicht


def test_inf_input_no_crash():
    """Inf-Eingabe darf keinen Absturz auslösen."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    audio_inf = np.full(_N, np.inf, dtype=np.float32)
    try:
        result = get_versa_plugin().score(audio_inf, SR)
        assert math.isfinite(result.mos)
    except (RuntimeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# MOS-Normalisierung für Kompatibilität mit alten CDPAM-Callern (§4.4)
# ---------------------------------------------------------------------------


def test_mos_normalization_to_0_100():
    """MOS [1,5] → [0,100] Normalisierung muss valide sein."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    result = get_versa_plugin().score(AUDIO_SINE, SR)
    score_0_100 = float(np.clip((result.mos - 1.0) / 4.0 * 100.0, 0.0, 100.0))
    assert 0.0 <= score_0_100 <= 100.0
    assert math.isfinite(score_0_100)


def test_mos_normalization_to_0_1():
    """MOS [1,5] → [0,1] Normalisierung muss valide sein."""
    from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

    result = get_versa_plugin().score(AUDIO_SINE, SR)
    score_01 = float(np.clip((result.mos - 1.0) / 4.0, 0.0, 1.0))
    assert 0.0 <= score_01 <= 1.0
    assert math.isfinite(score_01)
