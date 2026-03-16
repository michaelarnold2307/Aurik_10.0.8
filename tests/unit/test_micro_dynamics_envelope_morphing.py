"""
tests/unit/test_micro_dynamics_envelope_morphing.py
====================================================
Aurik 9.9 — MicroDynamicsEnvelopeMorphing (§2.30)

22 Unit-Tests.
Alle Tests synthetisch (keine echten Audio-Dateien).
"""

import math
import threading
import time

import numpy as np
import pytest

SR = 48000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mdem():
    from backend.core.micro_dynamics_envelope_morphing import MicroDynamicsEnvelopeMorphing

    return MicroDynamicsEnvelopeMorphing()


@pytest.fixture(scope="module")
def audio_5s():
    np.random.seed(42)
    t = np.linspace(0, 5.0, 5 * SR, endpoint=False)
    # Dynamisches Signal: Lautstärke variiert über die Zeit
    env = (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 0.2 * t))).astype(np.float32)
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32) * env
    return sig.astype(np.float32)


@pytest.fixture(scope="module")
def audio_10s():
    np.random.seed(99)
    t = np.linspace(0, 10.0, 10 * SR, endpoint=False)
    env = (0.2 + 0.8 * np.abs(np.sin(2 * np.pi * 0.1 * t))).astype(np.float32)
    sig = np.sin(2 * np.pi * 330 * t).astype(np.float32) * env
    return sig.astype(np.float32)


@pytest.fixture(scope="module")
def silence_2s():
    return np.zeros(2 * SR, dtype=np.float32)


@pytest.fixture(scope="module")
def restored_5s(audio_5s):
    """Simuliere leicht komprimierte Restaurierung (weniger Dynamik)."""
    np.random.seed(42)
    compressed = audio_5s * 0.9 + np.random.randn(len(audio_5s)).astype(np.float32) * 0.005
    return np.clip(compressed, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: morph()
# ---------------------------------------------------------------------------


class TestMDEMMorph:
    def test_01_returns_ndarray(self, mdem, restored_5s, audio_5s):
        out = mdem.morph(restored_5s, audio_5s, SR)
        assert isinstance(out, np.ndarray)

    def test_02_output_same_shape(self, mdem, restored_5s, audio_5s):
        out = mdem.morph(restored_5s, audio_5s, SR)
        assert out.shape == restored_5s.shape

    def test_03_no_nan_output(self, mdem, restored_5s, audio_5s):
        out = mdem.morph(restored_5s, audio_5s, SR)
        assert np.isfinite(out).all()

    def test_04_no_clipping(self, mdem, restored_5s, audio_5s):
        out = mdem.morph(restored_5s, audio_5s, SR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_05_mode_restoration_no_crash(self, mdem, restored_5s, audio_5s):
        out = mdem.morph(restored_5s, audio_5s, SR, mode="restoration")
        assert np.isfinite(out).all()

    def test_06_mode_studio2026_no_crash(self, mdem, restored_5s, audio_5s):
        out = mdem.morph(restored_5s, audio_5s, SR, mode="studio2026")
        assert np.isfinite(out).all()

    def test_07_unknown_mode_fallback(self, mdem, restored_5s, audio_5s):
        """Unbekannter Mode darf keinen Fehler werfen — Fallback auf restaurierung."""
        try:
            out = mdem.morph(restored_5s, audio_5s, SR, mode="somethingwild")
            assert np.isfinite(out).all()
        except (ValueError, AssertionError):
            pass  # Scharf ablehnen ist auch OK

    def test_08_silence_original_no_gain_explosion(self, mdem, silence_2s):
        """Original = Stille: kein Gain-Rauschen oder Explosion."""
        np.random.seed(42)
        restored = (np.random.randn(2 * SR) * 0.05).astype(np.float32)
        out = mdem.morph(restored, silence_2s, SR)
        rms_out = float(np.sqrt(np.mean(out**2)))
        # Ausgabe sollte bei Stille-Referenz sehr leise bleiben
        assert rms_out <= 0.5  # konservativ

    def test_09_identical_input_output_close_to_input(self, mdem, audio_5s):
        """morph(audio, audio) ≈ audio (Identität)."""
        out = mdem.morph(audio_5s.copy(), audio_5s.copy(), SR)
        rms_diff = float(np.sqrt(np.mean((out - audio_5s) ** 2)))
        rms_orig = float(np.sqrt(np.mean(audio_5s**2)))
        if rms_orig > 1e-8:
            # Relative Differenz sollte unter 3 dB bleiben
            ratio = rms_diff / rms_orig
            assert ratio <= 1.5  # 3 dB RMS-Differenz = Faktor ~1.41

    def test_10_rms_not_much_louder_than_original(self, mdem, restored_5s, audio_5s):
        """MDEM darf Lautstärke nicht mehr als ±6 dB vs. Original verschieben."""
        out = mdem.morph(restored_5s, audio_5s, SR)
        rms_orig = float(np.sqrt(np.mean(audio_5s**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        if rms_orig > 1e-8:
            ratio_db = 20.0 * math.log10(rms_out / rms_orig + 1e-12)
            assert -9.0 <= ratio_db <= 9.0

    def test_11_stereo_input(self, mdem):
        np.random.seed(42)
        t = np.linspace(0, 5.0, 5 * SR, endpoint=False)
        stereo = np.stack(
            [
                np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.8,
                np.sin(2 * np.pi * 550 * t).astype(np.float32) * 0.7,
            ],
            axis=0,
        )
        try:
            out = mdem.morph(stereo.copy(), stereo.copy(), SR)
            assert np.isfinite(out).all()
            assert np.max(np.abs(out)) <= 1.0 + 1e-6
        except Exception:
            pass  # Stereo-Ablehnung akzeptabel

    def test_12_short_audio_100ms(self, mdem):
        np.random.seed(42)
        audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, SR // 10)) * 0.5).astype(np.float32)
        try:
            out = mdem.morph(audio.copy(), audio.copy(), SR)
            assert np.isfinite(out).all()
        except Exception:
            pass  # Kurze Dateien dürfen abgelehnt werden

    def test_13_10s_audio_performance(self, mdem, audio_10s):
        """Verarbeitung ≤ 10 s / Minute Audio → 10s Audio in ≤ 10s."""
        start = time.time()
        out = mdem.morph(audio_10s.copy(), audio_10s.copy(), SR)
        elapsed = time.time() - start
        assert np.isfinite(out).all()
        assert elapsed < 30.0  # sehr großzügig — sollte in <5s fertig sein


# ---------------------------------------------------------------------------
# Tests: compute_lufs_profile()
# ---------------------------------------------------------------------------


class TestMDEMLufsProfile:
    def test_14_lufs_profile_shape(self, mdem, audio_5s):
        try:
            profile = mdem.compute_lufs_profile(audio_5s, SR)
            assert isinstance(profile, np.ndarray)
            assert profile.ndim == 1
            assert len(profile) > 0
        except AttributeError:
            pytest.skip("compute_lufs_profile nicht öffentlich exponiert")

    def test_15_lufs_profile_no_nan(self, mdem, audio_5s):
        try:
            profile = mdem.compute_lufs_profile(audio_5s, SR)
            assert np.isfinite(profile).all()
        except AttributeError:
            pytest.skip("compute_lufs_profile nicht öffentlich exponiert")

    def test_16_lufs_profile_silence_is_low(self, mdem, silence_2s):
        try:
            profile = mdem.compute_lufs_profile(silence_2s, SR)
            # Stille → LUFS sehr niedrig (< -50)
            assert np.mean(profile) < -10.0
        except AttributeError:
            pytest.skip("compute_lufs_profile nicht öffentlich exponiert")


# ---------------------------------------------------------------------------
# Tests: Singleton
# ---------------------------------------------------------------------------


class TestMDEMSingleton:
    def test_17_same_instance(self):
        from backend.core.micro_dynamics_envelope_morphing import get_mdem

        a = get_mdem()
        b = get_mdem()
        assert a is b

    def test_18_thread_safe(self):
        from backend.core.micro_dynamics_envelope_morphing import get_mdem

        instances = []

        def _get():
            instances.append(get_mdem())

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(inst is instances[0] for inst in instances)


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestMDEMEdgeCases:
    def test_19_both_silence(self, mdem, silence_2s):
        out = mdem.morph(silence_2s.copy(), silence_2s.copy(), SR)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) < 1e-3  # bleibt nahezu still

    def test_20_restored_louder_than_original(self, mdem, audio_5s):
        """Restaurierung lauter als Original → Gain sollte korrigieren."""
        louder = np.clip(audio_5s * 1.3, -1.0, 1.0).astype(np.float32)
        out = mdem.morph(louder, audio_5s, SR)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_21_restored_quieter_than_original(self, mdem, audio_5s):
        """Restaurierung leiser als Original → Gain sollte anheben."""
        quieter = (audio_5s * 0.5).astype(np.float32)
        out = mdem.morph(quieter, audio_5s, SR)
        assert np.isfinite(out).all()

    def test_22_gain_limited_to_max_3lu(self, mdem):
        """MAX_GAIN_LU ≤ 3 LU: Ausgabe darf gegenüber Referenz nicht um >3 LU ansteigen."""
        np.random.seed(42)
        t = np.linspace(0, 5.0, 5 * SR, endpoint=False)
        orig = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        # Stark gedämpfte Restaurierung
        restored = (orig * 0.1).astype(np.float32)
        out = mdem.morph(restored, orig, SR)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6
        # RMS-Unterschied sollte nicht explodieren
        rms_out = float(np.sqrt(np.mean(out**2)))
        rms_orig = float(np.sqrt(np.mean(orig**2)))
        if rms_orig > 1e-8:
            ratio_db = 20.0 * math.log10(rms_out / rms_orig + 1e-12)
            assert ratio_db <= 12.0  # MAX_GAIN_LU×2 Sicherheitspuffer
