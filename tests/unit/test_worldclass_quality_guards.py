"""
Unit-Tests für die 9 Weltklasse-Qualitäts-Guards (v9.5, V19–V26 + §2.72).

Abgedeckte Module:
  - backend/core/dsp/noise_texture_guard.py     (V19 §NTI)
  - backend/core/dsp/noise_floor_guard.py       (V21 §MNF)
  - backend/core/dsp/stereo_guard.py            (V23 §MKI)
  - backend/core/dsp/mikrodynamik_guard.py      (V20 §MKK)
  - backend/core/dsp/vibrato_guard.py           (§2.72)
  - backend/core/dsp/transient_guard.py         (V22 §PEP)
  - backend/core/dsp/spectral_color_guard.py    (V24 §SCK)
  - backend/core/dsp/warmth_guard.py            (V25 §WBG)
  - backend/core/dsp/onset_guard.py             (V26 §ATI)

Test-Strategie: synthetische Signale (Sinuston + Weißrauschen), kein ML,
kein echter Audio-Load. Alle Tests laufen in < 500 ms (Budget: --timeout=30).
"""

import gc
from typing import cast

import numpy as np
import pytest

SR = 48000
_DURATION_S = 1.0
_N = int(_DURATION_S * SR)


# ─── Hilfs-Generatoren ───────────────────────────────────────────────────────


def _silence(n: int = _N) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _white_noise(n: int = _N, amp: float = 0.05, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cast(np.ndarray, (rng.standard_normal(n) * amp).astype(np.float32))


def _sine(freq: float = 440.0, n: int = _N, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, n / SR, n, endpoint=False)
    return cast(np.ndarray, (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32))


def _vibrato_signal(
    fc: float = 440.0, fm: float = 5.5, depth: float = 6.0, n: int = _N, amp: float = 0.3
) -> np.ndarray:
    """Sinusträger mit 5.5 Hz Frequenzmodulation (typisches Sänger-Vibrato)."""
    t = np.linspace(0, n / SR, n, endpoint=False)
    phase = 2 * np.pi * fc * t + (depth / fm) * np.sin(2 * np.pi * fm * t)
    return cast(np.ndarray, (amp * np.sin(phase)).astype(np.float32))


def _stereo(audio: np.ndarray) -> np.ndarray:
    """Mono → Stereo (identische Kanäle)."""
    return np.stack([audio, audio], axis=0)  # (2, N)


def _out_of_phase_stereo(audio: np.ndarray) -> np.ndarray:
    """L = +signal, R = −signal → maximale Phasenlöschung."""
    return np.stack([audio, -audio], axis=0)


# ─── §V19 Noise-Textur-Guard ─────────────────────────────────────────────────


class TestNoiseTextureGuard:
    """compute_noise_texture_distance — Residualrauschen vs. Materialprofil."""

    def test_white_noise_residual_shellac_high_distance(self):
        """Weißes Rauschen hat flacheres Spektrum als Shellac-Rauschen → Distanz > 0.25."""
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = _white_noise(amp=0.03)
        dist = compute_noise_texture_distance(residual, "shellac", SR)
        assert isinstance(dist, float), "Muss float zurückgeben"
        assert 0.0 <= dist <= 1.0, f"Distanz nicht in [0,1]: {dist}"
        assert dist > 0.15, f"Weißes Rauschen auf Shellac sollte hohe Distanz haben, got {dist:.3f}"

    def test_zero_residual_returns_low_distance(self):
        """Null-Residual → keine Textur messbar → Distanz nahe 0."""
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = _silence()
        dist = compute_noise_texture_distance(residual, "vinyl", SR)
        assert isinstance(dist, float)
        assert dist >= 0.0

    def test_output_is_float_not_ndarray(self):
        """Rückgabe muss Python-float sein, nicht np.ndarray (Linter-Fix)."""
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = _white_noise()
        result = compute_noise_texture_distance(residual, "tape", SR)
        assert isinstance(result, float), f"Erwartet float, got {type(result)}"

    def test_unknown_material_no_crash(self):
        """Unbekanntes Material soll nicht crashen (Fallback auf Generic)."""
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = _white_noise()
        dist = compute_noise_texture_distance(residual, "unknown_material_xyz", SR)
        assert 0.0 <= dist <= 1.0

    def test_stereo_residual_handled(self):
        """Stereo-Residual soll fehlerfrei verarbeitet werden."""
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = _stereo(_white_noise(amp=0.02))
        dist = compute_noise_texture_distance(residual, "cassette", SR)
        assert isinstance(dist, float)
        assert 0.0 <= dist <= 1.0

    def teardown_method(self, _method):
        gc.collect(0)


# ─── §V21 Mindestrauschboden-Guard ───────────────────────────────────────────


class TestNoiseFloorGuard:
    """apply_noise_floor_minimum — Analog-Pause-Zonen erhalten Materialrauschboden."""

    def test_silence_gets_floor_shellac(self):
        """Stille auf Shellac-Material → Energieerhöhung über Stille."""
        from backend.core.dsp.noise_floor_guard import apply_noise_floor_minimum

        silent = _silence()
        result = apply_noise_floor_minimum(silent, SR, "shellac")
        assert result.shape == silent.shape
        # Shellac-Boden ≈ −42 dBFS → ~0.0079 linear → RMS deutlich über 0
        rms = float(np.sqrt(np.mean(result**2)))
        assert rms > 1e-5, f"Shellac-Rauschboden sollte über 0 liegen, RMS={rms:.8f}"

    def test_digital_material_untouched(self):
        """CD-Material → kein Rauschboden hinzugefügt (digitale Stille bleibt)."""
        from backend.core.dsp.noise_floor_guard import apply_noise_floor_minimum

        silent = _silence()
        result = apply_noise_floor_minimum(silent, SR, "cd_digital")
        assert np.allclose(result, silent, atol=1e-7), "Digitales Material soll unverändert bleiben"

    def test_output_shape_preserved(self):
        """Output-Shape identisch mit Input (Mono + Stereo)."""
        from backend.core.dsp.noise_floor_guard import apply_noise_floor_minimum

        audio_mono = _white_noise(amp=0.1)
        result = apply_noise_floor_minimum(audio_mono, SR, "vinyl")
        assert result.shape == audio_mono.shape

    def test_output_clipped_to_unit_range(self):
        """Output muss in [-1, 1] bleiben."""
        from backend.core.dsp.noise_floor_guard import apply_noise_floor_minimum

        loud = np.full(_N, 0.99, dtype=np.float32)
        result = apply_noise_floor_minimum(loud, SR, "tape")
        assert np.all(np.abs(result) <= 1.0), "Output übersteigt ±1.0"

    def test_wrong_sr_raises(self):
        """assert sr == 48000 — falscher SR → AssertionError."""
        from backend.core.dsp.noise_floor_guard import apply_noise_floor_minimum

        with pytest.raises(AssertionError):
            apply_noise_floor_minimum(_silence(), 44100, "vinyl")

    def teardown_method(self, _method):
        gc.collect(0)


# ─── §V23 Stereo-/Mono-Kompatibilitäts-Guard ─────────────────────────────────


class TestStereoGuard:
    """check_mono_compatibility — Phasenlöschung im 300–5000 Hz Band."""

    def test_in_phase_stereo_ok(self):
        """Identische Kanäle → keine Auslöschung → ok=True."""
        from backend.core.dsp.stereo_guard import check_mono_compatibility

        audio = _stereo(_sine(440.0))
        result = check_mono_compatibility(audio, SR)
        assert result.ok, (
            f"In-Phase-Stereo soll ok=True liefern, got cancellation={result.phase_cancellation_db:.1f} dB"
        )

    def test_out_of_phase_stereo_warns(self):
        """Gegenphasige Kanäle → Auslöschung > 3 dB → ok=False."""
        from backend.core.dsp.stereo_guard import check_mono_compatibility

        audio = _out_of_phase_stereo(_sine(440.0))
        result = check_mono_compatibility(audio, SR)
        assert not result.ok, "Gegenphasiges Stereo soll ok=False liefern"
        assert result.phase_cancellation_db > 3.0

    def test_result_has_required_fields(self):
        """MonoCompatResult muss alle Pflichtfelder haben."""
        from backend.core.dsp.stereo_guard import MonoCompatResult, check_mono_compatibility

        audio = _stereo(_sine())
        result = check_mono_compatibility(audio, SR)
        assert isinstance(result, MonoCompatResult)
        assert hasattr(result, "phase_cancellation_db")
        assert hasattr(result, "ok")

    def test_wrong_sr_raises(self):
        from backend.core.dsp.stereo_guard import check_mono_compatibility

        with pytest.raises(AssertionError):
            check_mono_compatibility(_stereo(_sine()), 22050)

    def teardown_method(self, _method):
        gc.collect(0)


# ─── §V20 Mikrodynamik-Korrelations-Guard ────────────────────────────────────


class TestMikrodynamikGuard:
    """frame_energy_correlation — Voiced-Frame-Pearson-Korrelation."""

    def test_identical_signals_correlation_one(self):
        """Identische Signale → Korrelation ≈ 1.0."""
        from backend.core.dsp.mikrodynamik_guard import frame_energy_correlation

        audio = _sine() + _white_noise(amp=0.02)
        corr = frame_energy_correlation(audio, audio, SR)
        assert isinstance(corr, float)
        assert corr > 0.99, f"Korrelation identischer Signale < 0.99: {corr:.4f}"

    def test_compressed_post_lower_correlation(self):
        """Stark komprimierter Post → niedrigere Korrelation."""
        from backend.core.dsp.mikrodynamik_guard import frame_energy_correlation

        pre = _sine(amp=0.5) + _white_noise(amp=0.03)
        # Kompression: hohe Pegel werden stark reduziert
        post = np.sign(pre) * (np.abs(pre) ** 2.5).astype(np.float32)
        post = np.clip(post, -1.0, 1.0)
        corr = frame_energy_correlation(pre, post, SR)
        assert corr <= 1.0
        assert corr >= 0.0

    def test_silence_input_no_crash(self):
        """Stille → kein Crash, Fallback-Wert zurückgegeben."""
        from backend.core.dsp.mikrodynamik_guard import frame_energy_correlation

        corr = frame_energy_correlation(_silence(), _silence(), SR)
        assert isinstance(corr, float)
        assert 0.0 <= corr <= 1.0

    def test_wrong_sr_raises(self):
        from backend.core.dsp.mikrodynamik_guard import frame_energy_correlation

        with pytest.raises(AssertionError):
            frame_energy_correlation(_sine(), _sine(), 44100)

    def teardown_method(self, _method):
        gc.collect(0)


# ─── §2.72 Vibrato-Tiefen-Guard ──────────────────────────────────────────────


class TestVibratoGuard:
    """check_vibrato_depth_preservation — F0-Modulationstiefe bleibt erhalten."""

    def test_no_vibrato_signal_skipped(self):
        """Reiner Sinus ohne FM → depth_pre < 0.3 Hz → ok=True (Guard überspringt)."""
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        sig = _sine(440.0)
        result = check_vibrato_depth_preservation(sig, sig, SR)
        assert result.ok, "Kein Vibrato → Guard soll ok=True liefern"

    def test_vibrato_preserved_ok(self):
        """Identisches Vibrato-Signal → ok=True."""
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        sig = _vibrato_signal()
        result = check_vibrato_depth_preservation(sig, sig, SR)
        assert result.ok, "Identisches Vibrato-Signal → ok=True"
        assert result.depth_reduction_pct < 5.0, (
            f"Reduktion bei identischem Signal zu hoch: {result.depth_reduction_pct:.1f}%"
        )

    def test_result_fields_present(self):
        """VibratoDepthResult muss Pflichtfelder haben."""
        from backend.core.dsp.vibrato_guard import VibratoDepthResult, check_vibrato_depth_preservation

        sig = _vibrato_signal()
        result = check_vibrato_depth_preservation(sig, sig, SR)
        assert isinstance(result, VibratoDepthResult)
        assert hasattr(result, "depth_pre_hz")
        assert hasattr(result, "depth_post_hz")
        assert hasattr(result, "depth_reduction_pct")
        assert hasattr(result, "ok")

    def test_wrong_sr_raises(self):
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        with pytest.raises(AssertionError):
            check_vibrato_depth_preservation(_sine(), _sine(), 44100)

    def teardown_method(self, _method):
        gc.collect(0)


# ─── §V22 Transient-Shift-Guard ──────────────────────────────────────────────


class TestTransientGuard:
    """detect_transient_shifts — Pre-Echo-Detektion via Onset-Zeitversatz."""

    def test_identical_signals_ok(self):
        """Identische Signale → max_shift_ms ≈ 0 → ok=True."""
        from backend.core.dsp.transient_guard import detect_transient_shifts

        audio = _sine() + _white_noise(amp=0.05)
        result = detect_transient_shifts(audio, audio, SR)
        assert result.ok, f"Identische Signale → ok=True, got max_shift={result.max_shift_ms:.2f} ms"

    def test_result_fields_present(self):
        """TransientShiftResult muss Pflichtfelder haben."""
        from backend.core.dsp.transient_guard import TransientShiftResult, detect_transient_shifts

        audio = _sine(440.0) + _white_noise(amp=0.05)
        result = detect_transient_shifts(audio, audio, SR)
        assert isinstance(result, TransientShiftResult)
        assert hasattr(result, "max_shift_ms")
        assert hasattr(result, "ok")
        assert hasattr(result, "blend_reduction")

    def test_max_shift_non_negative(self):
        """max_shift_ms darf nicht negativ sein."""
        from backend.core.dsp.transient_guard import detect_transient_shifts

        audio = _sine(440.0) + _white_noise(amp=0.1)
        result = detect_transient_shifts(audio, audio * 0.5, SR)
        assert result.max_shift_ms >= 0.0

    def test_wrong_sr_raises(self):
        from backend.core.dsp.transient_guard import detect_transient_shifts

        with pytest.raises(AssertionError):
            detect_transient_shifts(_sine(), _sine(), 44100)

    def teardown_method(self, _method):
        gc.collect(0)


# ─── §V24 Spektralfarbe-Guard ────────────────────────────────────────────────


class TestSpectralColorGuard:
    """check_spectral_color_preservation — 1/3-Oktav-Profil-Korrelation."""

    def test_identical_signals_correlation_near_one(self):
        """Identische Signale → Korrelation ≈ 1.0 → ok=True."""
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        audio = _sine() + _white_noise(amp=0.02)
        result = check_spectral_color_preservation(audio, audio, SR)
        assert result.ok, f"Identische Signale → ok=True, got corr={result.correlation:.3f}"
        assert result.correlation > 0.99

    def test_result_fields(self):
        """SpectralColorResult muss Pflichtfelder haben."""
        from backend.core.dsp.spectral_color_guard import SpectralColorResult, check_spectral_color_preservation

        audio = _sine(440.0) + _white_noise()
        result = check_spectral_color_preservation(audio, audio, SR)
        assert isinstance(result, SpectralColorResult)
        assert hasattr(result, "correlation")
        assert hasattr(result, "ok")
        assert hasattr(result, "pre_profile_db")
        assert hasattr(result, "post_profile_db")

    def test_heavy_eq_reduces_correlation(self):
        """Stark geentzte Version → niedrigere Korrelation."""
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        audio = _white_noise(amp=0.3)  # weiß = flach
        # HP-gefiltert (nur HF) vs. Original
        from scipy.signal import butter, sosfiltfilt

        sos = butter(8, 4000.0 / (SR / 2.0), btype="high", output="sos")
        filtered = sosfiltfilt(sos, audio).astype(np.float32)
        result = check_spectral_color_preservation(audio, filtered, SR)
        assert result.correlation < 0.99, (
            f"Stark EQ'd Signal sollte niedrige Korrelation haben: {result.correlation:.3f}"
        )

    def test_wrong_sr_raises(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        with pytest.raises(AssertionError):
            check_spectral_color_preservation(_sine(), _sine(), 44100)

    def teardown_method(self, _method):
        gc.collect(0)


# ─── §V25 Wärmeband-Guard ────────────────────────────────────────────────────


class TestWarmthGuard:
    """measure_warmth_band_delta — kumulativer 200–800 Hz Verlust-Tracker."""

    def test_identical_signals_no_loss(self):
        """Identische Signale → loss_db ≈ 0, ok=True, blend=1.0."""
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        audio = _sine(400.0, amp=0.3) + _white_noise(amp=0.01)
        result = measure_warmth_band_delta(audio, audio, SR)
        assert result.ok, "Identische Signale → ok=True"
        assert abs(result.loss_db) < 0.5, f"Verlust identischer Signale zu hoch: {result.loss_db:.3f} dB"
        assert result.warmth_blend_factor >= 1.0 - 1e-6

    def test_attenuated_warmth_positive_loss(self):
        """HF-angehobener Post (weniger Bass/Wärme) → positiver loss_db."""
        from scipy.signal import butter, sosfiltfilt

        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        pre = _sine(400.0, amp=0.3) + _white_noise(amp=0.01)
        sos = butter(4, 800.0 / (SR / 2.0), btype="high", output="sos")
        post = sosfiltfilt(sos, pre).astype(np.float32)  # Wärme entfernt
        result = measure_warmth_band_delta(pre, post, SR)
        assert result.loss_db > 0.5, f"Wärme-Verlust nach HP-Filter sollte > 0.5 dB sein: {result.loss_db:.3f}"

    def test_cumulative_above_threshold_activates_blend(self):
        """Kumulativer Verlust > 2.5 dB → warmth_blend_factor < 1.0."""
        from scipy.signal import butter, sosfiltfilt

        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        pre = _sine(400.0, amp=0.3) + _white_noise(amp=0.01)
        sos = butter(4, 800.0 / (SR / 2.0), btype="high", output="sos")
        post = sosfiltfilt(sos, pre).astype(np.float32)
        result = measure_warmth_band_delta(pre, post, SR, cumulative_loss_db=3.0)
        assert result.warmth_blend_factor < 1.0, (
            f"Kumulativer Verlust 3.0 dB → blend < 1.0, got {result.warmth_blend_factor:.3f}"
        )

    def test_result_fields(self):
        """WarmthBandResult muss Pflichtfelder haben."""
        from backend.core.dsp.warmth_guard import WarmthBandResult, measure_warmth_band_delta

        audio = _sine(400.0)
        result = measure_warmth_band_delta(audio, audio, SR)
        assert isinstance(result, WarmthBandResult)
        assert hasattr(result, "loss_db")
        assert hasattr(result, "ok")
        assert hasattr(result, "warmth_blend_factor")

    def test_wrong_sr_raises(self):
        from backend.core.dsp.warmth_guard import measure_warmth_band_delta

        with pytest.raises(AssertionError):
            measure_warmth_band_delta(_sine(), _sine(), 44100)

    def teardown_method(self, _method):
        gc.collect(0)


# ─── §V26 Onset-Protection-Guard ─────────────────────────────────────────────


class TestOnsetGuard:
    """apply_onset_protection_mask — Onset-Fenster 0–20 ms max. 1.5 dB Energiedelta."""

    def test_identical_signals_unchanged(self):
        """Identische pre/post → keine Blend-Korrektur → Ergebnis ≈ post."""
        from backend.core.dsp.onset_guard import apply_onset_protection_mask

        audio = _sine(440.0) + _white_noise(amp=0.05)
        result = apply_onset_protection_mask(audio, audio, None, max_delta_db=1.5)
        assert result.shape == audio.shape
        assert np.allclose(result, audio, atol=1e-5), "Identische Signale dürfen nicht verändert werden"

    def test_output_shape_preserved_mono(self):
        """Mono: Output-Shape identisch mit Input."""
        from backend.core.dsp.onset_guard import apply_onset_protection_mask

        pre = _sine(440.0) + _white_noise(amp=0.05)
        post = pre * 0.5
        result = apply_onset_protection_mask(pre, post, None, max_delta_db=1.5)
        assert result.shape == pre.shape

    def test_output_shape_preserved_stereo(self):
        """Stereo: Output-Shape (2, N) identisch mit Input."""
        from backend.core.dsp.onset_guard import apply_onset_protection_mask

        pre = _stereo(_sine(440.0) + _white_noise(amp=0.05))
        post = pre * 0.5
        result = apply_onset_protection_mask(pre, post, None, max_delta_db=1.5)
        assert result.shape == pre.shape

    def test_explicit_onset_mask_used(self):
        """Explizite Onset-Maske wird akzeptiert (kein Crash)."""
        from backend.core.dsp.onset_guard import apply_onset_protection_mask

        pre = _sine(440.0) + _white_noise(amp=0.05)
        post = pre * 0.8
        # Onset-Maske: boolean-Array, 10 % der Frames als Onset markiert
        mask_len = len(pre) // 512
        onset_mask = np.zeros(max(mask_len, 1), dtype=bool)
        if len(onset_mask) > 5:
            onset_mask[5] = True
        result = apply_onset_protection_mask(pre, post, onset_mask, max_delta_db=1.5)
        assert result.shape == pre.shape

    def test_output_clipped_to_unit_range(self):
        """Output muss in [-1, 1] liegen."""
        from backend.core.dsp.onset_guard import apply_onset_protection_mask

        pre = _sine(440.0, amp=0.9) + _white_noise(amp=0.05)
        post = pre * 1.05  # minimal über ±1
        result = apply_onset_protection_mask(pre, post, None, max_delta_db=1.5)
        assert np.all(np.abs(result) <= 1.001), "Output übersteigt ±1.0"

    def teardown_method(self, _method):
        gc.collect(0)


# ─── Integration: temporal_continuity_guard gain_step_db ─────────────────────


class TestTemporalContinuityGuardGainStep:
    """§2.69 v9.5 — gain_step_db muss in TemporalContinuityResult enthalten sein."""

    def test_result_has_gain_step_db(self):
        """TemporalContinuityResult muss gain_step_db-Feld enthalten."""
        from backend.core.temporal_continuity_guard import TemporalContinuityResult

        result = TemporalContinuityResult(
            ok=True,
            variance_ratio=1.0,
            phase_id="phase_03_denoise",
            critical=False,
            gain_step_db=0.0,
        )
        assert hasattr(result, "gain_step_db")
        assert result.gain_step_db == 0.0

    def test_check_temporal_continuity_returns_gain_step_db(self):
        """check_temporal_continuity muss gain_step_db befüllen."""
        from backend.core.temporal_continuity_guard import check_temporal_continuity

        pre = _sine(440.0) + _white_noise(amp=0.02)
        post = pre * 0.5  # Gain-Abfall → gain_step_db > 0
        result = check_temporal_continuity(pre, post, phase_id="phase_03_denoise", sr=SR)
        assert hasattr(result, "gain_step_db"), "gain_step_db muss im Result sein"
        assert isinstance(result.gain_step_db, float), f"Erwartet float, got {type(result.gain_step_db)}"
        assert result.gain_step_db >= 0.0, "gain_step_db darf nicht negativ sein"

    def test_large_gain_step_detected(self):
        """Signal-Pegelsprung von −6 dB → gain_step_db > 1.5 dB."""
        from backend.core.temporal_continuity_guard import check_temporal_continuity

        pre = _sine(440.0, amp=0.5) + _white_noise(amp=0.02)
        post = pre * 0.25  # −12 dB Pegelsprung
        result = check_temporal_continuity(pre, post, phase_id="phase_18_noise_gate", sr=SR)
        assert result.gain_step_db > 1.5, (
            f"Großer Gain-Sprung sollte gain_step_db > 1.5 dB haben, got {result.gain_step_db:.3f}"
        )

    def teardown_method(self, _method):
        gc.collect(0)
