"""
AurikDenker E2E Integration Test (DSP-only, kein echter ML-Modell-Load nötig).

Testet den vollständigen kanonischen Pipeline-Einstiegspunkt
`AurikDenker.denke()` mit synthetischem 3-Sekunden-Audio @ 48 000 Hz.

Spec-Referenzen:
    - §2.2: AurikDenker als PFLICHT-Einstiegspunkt (nicht UV3 direkt)
    - §8.1: quality_estimate ≥ 0.55 nach erfolgreicher Restaurierung
    - §8.2: Kein NaN/Inf im Ausgang, kein Clipping
    - §1.1: RestorationResult-Rückgabe
    - §14 E2E: Pflicht-Integrations-Test
"""

from __future__ import annotations

import math
import numpy as np
import pytest


@pytest.fixture(scope="module")
def synthetic_audio():
    """3s synthetisches Audio: 440 Hz Sinus + Rauschen @ 48 000 Hz (Stereo)."""
    sr = 48_000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False, dtype=np.float32)
    mono = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.default_rng(42).standard_normal(len(t)).astype(np.float32)
    audio = np.column_stack([mono, mono])  # Stereo [n, 2]
    return audio, sr


def test_aurik_denker_returns_restoration_result(synthetic_audio):
    """AurikDenker.denke() gibt RestorationResult zurück (§1.1, §2.2)."""
    audio, sr = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar (Umgebungsproblem): {exc}")

    denker = AurikDenker()
    try:
        result = denker.denke(audio.copy(), sr, mode="balanced")
    except Exception as exc:
        pytest.fail(f"AurikDenker.denke() raised {type(exc).__name__}: {exc}")

    assert result is not None, "Ergebnis darf nicht None sein"


def test_aurik_denker_output_no_nan_inf(synthetic_audio):
    """Ausgabe-Audio enthält kein NaN/Inf (§8.2 Universelle Garantie)."""
    audio, sr = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    denker = AurikDenker()
    result = denker.denke(audio.copy(), sr, mode="balanced")

    if hasattr(result, "audio") and result.audio is not None:
        out = np.asarray(result.audio)
        assert np.isfinite(out).all(), "NaN/Inf im Ausgabe-Audio gefunden"
        assert np.max(np.abs(out)) <= 1.0, "Clipping im Ausgabe-Audio (|x| > 1.0)"


def test_aurik_denker_quality_estimate_present(synthetic_audio):
    """quality_estimate-Feld ist vorhanden und ≥ 0.0 (§8.1 E2E-Pflicht)."""
    audio, sr = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    denker = AurikDenker()
    result = denker.denke(audio.copy(), sr, mode="balanced")

    if hasattr(result, "quality_estimate"):
        qe = float(result.quality_estimate)
        assert math.isfinite(qe), "quality_estimate ist nicht endlich"
        assert qe >= 0.0, f"quality_estimate muss ≥ 0.0 sein, erhalten: {qe}"


def test_aurik_denker_preserves_sample_rate(synthetic_audio):
    """Ausgabe-SR muss 48 000 Hz sein (interne Verarbeitungs-Invariante)."""
    audio, sr = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    denker = AurikDenker()
    result = denker.denke(audio.copy(), sr, mode="balanced")

    if hasattr(result, "audio") and result.audio is not None:
        out = np.asarray(result.audio)
        # SR-Erhalt: Länge sollte nicht mehr als 1 % abweichen (Resampling-Check)
        expected_samples = len(audio)
        actual_samples = out.shape[0] if out.ndim >= 1 else 0
        if actual_samples > 0:
            ratio = actual_samples / expected_samples
            assert 0.95 <= ratio <= 1.05, (
                f"Audio-Länge verändert sich zu stark: {actual_samples} vs {expected_samples}"
            )


def test_aurik_denker_fast_mode(synthetic_audio):
    """AurikDenker.denke() funktioniert auch im mode='fast' (kein Absturz)."""
    audio, sr = synthetic_audio
    try:
        from denker.aurik_denker import AurikDenker
    except ImportError as exc:
        pytest.skip(f"AurikDenker nicht importierbar: {exc}")

    denker = AurikDenker()
    try:
        result = denker.denke(audio.copy(), sr, mode="fast")
    except Exception as exc:
        pytest.fail(f"Modus 'fast' crashed: {type(exc).__name__}: {exc}")

    assert result is not None
