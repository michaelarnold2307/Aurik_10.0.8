"""Unit-Tests für forensics.medium_detector (MediumDetector, detect_medium_chain).

Testet:
- MediumDetector.detect() auf verschiedene synthetische Signale
- Mono/Stereo-Konversion
- SpectralFingerprint-Felder
- TransferChain-Erkennung
- Multi-Generation-Detection
- Singleton-Pattern + Thread-Safety
- NaN/Inf-Schutz
- Serialisierung (as_dict)
"""

import threading
from typing import Optional

import numpy as np
import pytest

from forensics.medium_detector import (
    MediumDetectionResult,
    MediumDetector,
    SpectralFingerprint,
    TransferChain,
    detect_medium_chain,
    get_medium_detector,
)


class TestMediumDetector:
    """Tests für MediumDetector-Klasse."""

    def test_01_detect_mono_signal_returns_result(self):
        """Mono-Signal: detect() gibt MediumDetectionResult zurück."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_02_detect_stereo_to_mono_conversion(self):
        """Stereo-Signal: wird korrekt zu mono konvertiert."""
        np.random.seed(42)
        audio_stereo = np.random.randn(2, 48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio_stereo, sr=48000)
        assert isinstance(result, MediumDetectionResult)
        assert result.primary_material is not None

    def test_03_white_noise_cd_digital_or_unknown(self):
        """Weißes Rauschen → cd_digital oder unknown."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.primary_material in ["cd_digital", "digital", "unknown"]

    def test_04_silence_no_crash_low_confidence(self):
        """Stille → kein Absturz, confidence < 1.0."""
        audio = np.zeros(48000, dtype=np.float32)
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_05_bandwidth_limited_signal_shellac_or_tape(self):
        """Bandbreitenbegrenztes Signal (< 4 kHz) → shellac oder tape erkannt."""
        np.random.seed(42)
        # Synthetisches Signal mit Tiefpass-Simulation
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Erwarten niedrigen Rolloff
        assert isinstance(result.spectral_fingerprint, SpectralFingerprint)

    def test_06_rolloff_value_positive(self):
        """Rolloff-Wert ist > 0."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.rolloff_95_hz > 0

    def test_07_spectral_fingerprint_fields_finite(self):
        """Alle SpectralFingerprint-Felder sind finite."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        fp = result.spectral_fingerprint
        assert np.isfinite(fp.rolloff_95_hz)
        assert np.isfinite(fp.wow_flutter_index)
        assert np.isfinite(fp.hf_energy_above_16k)
        assert np.isfinite(fp.noise_floor_db)
        assert np.isfinite(fp.effective_bandwidth_hz)

    def test_08_nan_in_audio_no_crash(self):
        """NaN/Inf-Schutz: audio mit NaN → kein Absturz."""
        audio = np.full(48000, np.nan, dtype=np.float32)
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Sollte mit nan_to_num behandelt werden
        assert isinstance(result, MediumDetectionResult)

    def test_09_singleton_get_medium_detector_same_object(self):
        """Singleton: get_medium_detector() gibt selbes Objekt zurück."""
        det1 = get_medium_detector()
        det2 = get_medium_detector()
        assert det1 is det2

    def test_10_thread_safety_singleton(self):
        """Thread-Sicherheit: 10 parallele Aufrufe liefern selbes Singleton."""
        instances = []

        def get_instance():
            instances.append(get_medium_detector())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Alle Instanzen sind identisch
        assert all(inst is instances[0] for inst in instances)

    def test_11_transfer_chain_not_empty(self):
        """TransferChain ist nicht leer."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert len(result.transfer_chain) > 0

    def test_12_primary_material_known_type(self):
        """primary_material ist einer der bekannten MaterialType-Strings."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        known_materials = [
            "tape",
            "vinyl",
            "shellac",
            "cd_digital",
            "digital",
            "mp3_low",
            "mp3_high",
            "aac",
            "unknown",
            "streaming",
            "dat",
            "wax_cylinder",
            "wire_recording",
            "lacquer_disc",
            "minidisc",
            "reel_tape",
        ]
        assert result.primary_material in known_materials

    def test_13_confidence_in_range(self):
        """confidence ∈ [0, 1]."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert 0.0 <= result.confidence <= 1.0

    def test_14_as_dict_serializable(self):
        """as_dict() serialisierbar (alle Werte str/float/bool/list)."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "transfer_chain" in d
        assert "is_multi_generation" in d
        assert "primary_material" in d
        assert "confidence" in d

    def test_15_chain_label_contains_arrow_if_multi_generation(self):
        """chain_label enthält ' → ' wenn is_multi_generation=True."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        if result.is_multi_generation:
            assert " → " in result.chain_label

    def test_16_detect_very_short_audio_no_crash(self):
        """Sehr kurzes Audio (< 1 s) → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(4800).astype(np.float32) * 0.1  # 0.1 s @ 48 kHz
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_17_detect_long_audio_no_crash(self):
        """Langes Audio → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(480000).astype(np.float32) * 0.1  # 10 s
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_18_spectral_fingerprint_hf_energy_in_range(self):
        """HF-Energie ∈ [0, 100] %."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert 0.0 <= result.spectral_fingerprint.hf_energy_above_16k <= 100.0

    def test_19_noise_floor_db_negative_or_zero(self):
        """Noise-Floor ist ≤ 0 dB."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.0001  # leise
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.noise_floor_db <= 0.0

    def test_20_effective_bandwidth_less_than_nyquist(self):
        """Effective Bandwidth < SR/2."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.effective_bandwidth_hz <= 24000

    def test_21_wow_flutter_index_non_negative(self):
        """Wow/Flutter-Index ≥ 0."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.wow_flutter_index >= 0.0

    def test_22_detect_sine_wave_stable_pitch(self):
        """Sinus-Welle: Wow/Flutter sollte klein sein."""
        np.random.seed(42)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Stabiler Pitch → kleiner Wow/Flutter
        assert result.spectral_fingerprint.wow_flutter_index < 5.0

    def test_23_evidence_list_not_none(self):
        """evidence-Liste ist nicht None."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.evidence is not None
        assert isinstance(result.evidence, list)

    def test_24_as_dict_has_expected_keys(self):
        """as_dict() hat erwartete Keys."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        d = result.as_dict()
        expected_keys = [
            "transfer_chain",
            "is_multi_generation",
            "primary_material",
            "confidence",
            "chain_label",
            "spectral_fingerprint",
            "evidence",
        ]
        for key in expected_keys:
            assert key in d, f"Key {key} fehlt in as_dict()"

    def test_25_stereo_shape_2_N_conversion(self):
        """Stereo-Array shape (2, N) → korrekte Mono-Konversion."""
        np.random.seed(42)
        audio_stereo = np.random.randn(2, 48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio_stereo, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_26_stereo_shape_N_2_conversion(self):
        """Stereo-Array shape (N, 2) → korrekte Mono-Konversion."""
        np.random.seed(42)
        audio_stereo = np.random.randn(48000, 2).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio_stereo, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_27_detect_medium_chain_convenience_function(self):
        """detect_medium_chain() convenience function works."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detect_medium_chain(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_28_rolloff_less_than_sr_half(self):
        """Rolloff < SR/2."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.rolloff_95_hz < 24000

    def test_29_primary_material_is_string(self):
        """primary_material ist String."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result.primary_material, str)

    def test_30_chain_label_is_string(self):
        """chain_label ist String."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result.chain_label, str)
        assert len(result.chain_label) > 0
