import numpy as np

from plugins.deepfilternet_v3_ii_plugin import _DF_ORDER, DeepFilterNetV3IIPlugin


def test_deepfilternet_v3_ii_plugin_aurik90():
    np.random.randn(16000).astype(np.float32)
    plugin = DeepFilterNetV3IIPlugin()
    assert isinstance(plugin, DeepFilterNetV3IIPlugin)


def test_apply_df_filter_vectorized_correctness():
    """Verify vectorized _apply_df_filter matches scalar reference implementation."""
    rng = np.random.RandomState(42)
    S, n_bins = 50, 96
    spec_cx = (rng.randn(481, S) + 1j * rng.randn(481, S)).astype(np.complex64)
    coefs = rng.randn(S, n_bins, _DF_ORDER).astype(np.float32)
    alpha = rng.rand(1, S, 1).astype(np.float32)

    # Scalar reference implementation (old code)
    result_ref = spec_cx.copy()
    for t in range(S):
        for b in range(n_bins):
            acc = complex(0.0)
            for k in range(_DF_ORDER):
                t_past = max(0, t - k)
                c = coefs[t, b, k]
                acc += c * spec_cx[b, t_past]
            blend = float(alpha[0, t, 0])
            result_ref[b, t] = blend * acc + (1 - blend) * spec_cx[b, t]

    # Vectorized implementation
    plugin = DeepFilterNetV3IIPlugin()
    result_vec = plugin._apply_df_filter(spec_cx.copy(), coefs, alpha)

    # Check numerical equivalence (allow small float tolerance)
    np.testing.assert_allclose(
        result_vec[:n_bins, :],
        result_ref[:n_bins, :],
        rtol=1e-4,
        atol=1e-6,
        err_msg="Vectorized _apply_df_filter deviates from scalar reference",
    )
    # Bins above n_bins must be unchanged
    np.testing.assert_array_equal(result_vec[n_bins:, :], spec_cx[n_bins:, :])


def test_compute_features_shape():
    """Verify _compute_features returns correct shapes for typical audio."""
    plugin = DeepFilterNetV3IIPlugin()
    mono = np.random.randn(48000).astype(np.float32)  # 1 second @ 48kHz
    feat_erb, feat_spec, spec_cx = plugin._compute_features(mono)
    assert feat_erb.ndim == 4 and feat_erb.shape[0] == 1 and feat_erb.shape[1] == 1
    assert feat_erb.shape[3] == 32  # n_erb
    assert feat_spec.ndim == 4 and feat_spec.shape[1] == 2
    assert feat_spec.shape[3] == 96  # DF_BINS
    assert spec_cx.shape[0] == 481  # N_FFT//2+1
    # S dimension must match
    assert feat_erb.shape[2] == spec_cx.shape[1]
    assert feat_spec.shape[2] == spec_cx.shape[1]


def test_compute_features_nan_free():
    """Verify _compute_features produces no NaN/Inf."""
    plugin = DeepFilterNetV3IIPlugin()
    mono = np.random.randn(96000).astype(np.float32)  # 2s
    feat_erb, feat_spec, spec_cx = plugin._compute_features(mono)
    assert np.all(np.isfinite(feat_erb))
    assert np.all(np.isfinite(feat_spec))
    assert np.all(np.isfinite(spec_cx))


def test_enhance_omlsa_fallback_short():
    """Verify OMLSA fallback works for short audio without ONNX models."""
    plugin = DeepFilterNetV3IIPlugin()
    plugin._enc = None  # Force OMLSA fallback
    audio = np.random.randn(48000).astype(np.float32) * 0.3  # 1s mono
    out = plugin.enhance(audio, sr=48000)
    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0


def test_enhance_stereo_omlsa_fallback():
    """Verify OMLSA fallback works for stereo audio."""
    plugin = DeepFilterNetV3IIPlugin()
    plugin._enc = None  # Force OMLSA fallback
    audio = np.random.randn(48000, 2).astype(np.float32) * 0.3
    out = plugin.enhance(audio, sr=48000)
    assert out.ndim == 2 and out.shape[1] == 2
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0
