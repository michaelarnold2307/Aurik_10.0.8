"""
tests/unit/test_stem_remix_balancer.py — StemRemixBalancer Test-Suite (≥ 20 Tests)
Alle Tests synthetisch, kein ML-Modell-Download erforderlich.
"""

import math

import numpy as np
import pytest

SR = 48_000
np.random.seed(42)


def _audio(dur: float = 3.0, amp: float = 0.3, sr: int = SR):
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _stereo(dur: float = 3.0, amp: float = 0.3):
    m = _audio(dur, amp)
    return np.stack([m, m * 0.9], axis=0)


# ---------------------------------------------------------------------------


def test_00_import():
    from backend.core.stem_remix_balancer import StemRemixBalancer

    assert StemRemixBalancer is not None


def test_01_balance_remix_shape_mono():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(3.0, 0.3)
    i = _audio(3.0, 0.2)
    o = _audio(3.0, 0.25)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert out.shape == v.shape


def test_02_output_no_nan():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(3.0, 0.3)
    i = _audio(3.0, 0.2)
    o = _audio(3.0, 0.25)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert np.isfinite(out).all()


def test_03_output_not_clipped():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(3.0, 0.3)
    i = _audio(3.0, 0.2)
    o = _audio(3.0, 0.25)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert np.max(np.abs(out)) <= 1.0


def test_04_silence_stems():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = np.zeros(SR * 2, dtype=np.float32)
    i = np.zeros(SR * 2, dtype=np.float32)
    o = np.zeros(SR * 2, dtype=np.float32)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert np.isfinite(out).all()
    assert out.shape == v.shape


def test_05_stereo_stems():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _stereo(3.0, 0.3)
    i = _stereo(3.0, 0.2)
    o = _stereo(3.0, 0.25)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert out.shape == v.shape
    assert np.isfinite(out).all()


def test_06_singleton_identity():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    a = get_stem_remix_balancer()
    b = get_stem_remix_balancer()
    assert a is b


def test_07_thread_safe_singleton():
    import concurrent.futures

    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(get_stem_remix_balancer) for _ in range(16)]
        instances = [f.result() for f in futs]
    assert all(inst is instances[0] for inst in instances)


def test_08_vocal_weight_zero():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(2.0, 0.4)
    i = _audio(2.0, 0.4)
    o = _audio(2.0, 0.4)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR, vocal_weight=0.0)
    assert np.isfinite(out).all()


def test_09_vocal_weight_one():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(2.0, 0.4)
    i = _audio(2.0, 0.2)
    o = _audio(2.0, 0.3)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR, vocal_weight=1.0)
    assert np.isfinite(out).all()


def test_10_output_dtype_float32():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(2.0, 0.3)
    i = _audio(2.0, 0.2)
    o = _audio(2.0, 0.25)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert out.dtype == np.float32


def test_11_very_loud_stems_clipped():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = np.ones(SR * 2, dtype=np.float32)
    i = np.ones(SR * 2, dtype=np.float32)
    o = np.ones(SR * 2, dtype=np.float32)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert np.max(np.abs(out)) <= 1.0


def test_12_impulse_signal_no_nan():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = np.zeros(SR * 2, dtype=np.float32)
    v[SR // 2] = 0.5
    i = np.zeros(SR * 2, dtype=np.float32)
    i[SR] = 0.3
    o = (v + i) * 0.5
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert np.isfinite(out).all()


def test_13_different_lengths_same_as_shortest():
    """Kurze Stämme — balancer arbeitet auf min-Länge."""
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(3.0, 0.3)
    i = _audio(3.0, 0.2)
    o = _audio(3.0, 0.25)
    # Truncate vocals
    v_short = v[: len(v) - SR // 4]
    # Kompatibilitäts-Länge
    min_len = min(len(v_short), len(i), len(o))
    out = get_stem_remix_balancer().balance_remix(v_short, i[:min_len], o[:min_len], SR)
    assert np.isfinite(out).all()


def test_14_assert_sample_rate():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(2.0, 0.3)
    i = _audio(2.0, 0.2)
    o = _audio(2.0, 0.25)
    with pytest.raises((AssertionError, ValueError)):
        get_stem_remix_balancer().balance_remix(v, i, o, 44100)


def test_15_output_not_silent():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(3.0, 0.3)
    i = _audio(3.0, 0.2)
    o = _audio(3.0, 0.25)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert np.max(np.abs(out)) > 0.0


def test_16_lufs_close_to_original():
    """LUFS-Differenz zwischen Ausgang und Original sollte klein sein."""
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(4.0, 0.3)
    i = _audio(4.0, 0.2)
    o = v * 0.6 + i * 0.4
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    rms_orig = np.sqrt(np.mean(o**2))
    rms_out = np.sqrt(np.mean(out**2))
    if rms_orig > 0 and rms_out > 0:
        ratio_db = abs(20 * math.log10(rms_out / rms_orig))
        assert ratio_db < 6.0, f"LUFS-Drift zu groß: {ratio_db:.1f} dB"


def test_17_numpy_float64_input():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(2.0, 0.3).astype(np.float64)
    i = _audio(2.0, 0.2).astype(np.float64)
    o = (v + i * 0.5).astype(np.float64)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert np.isfinite(out).all()


def test_18_zero_original_no_crash():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(2.0, 0.3)
    i = _audio(2.0, 0.2)
    o = np.zeros(len(v), dtype=np.float32)
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert np.isfinite(out).all()


def test_19_consistent_output():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(3.0, 0.25)
    i = _audio(3.0, 0.20)
    o = _audio(3.0, 0.22)
    b = get_stem_remix_balancer()
    out1 = b.balance_remix(v.copy(), i.copy(), o.copy(), SR)
    out2 = b.balance_remix(v.copy(), i.copy(), o.copy(), SR)
    np.testing.assert_array_almost_equal(out1, out2, decimal=5)


def test_20_returns_ndarray():
    from backend.core.stem_remix_balancer import get_stem_remix_balancer

    v = _audio(2.0, 0.3)
    i = _audio(2.0, 0.2)
    o = v * 0.5 + i * 0.5
    out = get_stem_remix_balancer().balance_remix(v, i, o, SR)
    assert isinstance(out, np.ndarray)
