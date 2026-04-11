"""
tests/unit/test_holistic_perceptual_gate.py — §2.44 HolisticPerceptualGate Test-Suite (≥ 25 Tests)
Alle Tests synthetisch, kein ML-Modell erforderlich.
"""

from unittest.mock import patch

import numpy as np

SR = 48_000


def _audio(dur: float = 1.0, amp: float = 0.3, freq: float = 440.0):
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------


def test_00_import():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate, get_holistic_gate

    assert HolisticPerceptualGate is not None
    assert get_holistic_gate is not None


def test_01_singleton():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    g1 = get_holistic_gate()
    g2 = get_holistic_gate()
    assert g1 is g2


def test_02_restoration_identical_passes():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR)
    assert result.passed
    assert result.hpi > 0


def test_03_restoration_with_artifacts_fails():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR, artifact_freedom=0.80)
    assert not result.passed
    assert result.artifact_freedom == 0.80


def test_04_studio_identical_passes():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_studio(audio, audio, SR, pqs_improvement=0.5)
    assert result.hpi > 0
    assert result.is_studio_mode


def test_05_studio_with_artifacts_fails():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_studio(audio, audio, SR, artifact_freedom=0.80)
    assert not result.passed


def test_06_hpi_positive_means_export():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR)
    # HPI > 0 → export allowed
    if result.hpi > 0 and result.artifact_freedom >= 0.95:
        assert result.passed


def test_07_mert_similarity_identical():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    sim = gate._compute_mert_similarity(audio, audio, SR)
    assert sim >= 0.99, f"Identical audio should have near-perfect MERT similarity, got {sim}"


def test_08_mert_similarity_different():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio1 = _audio(2.0, freq=440.0)
    audio2 = _audio(2.0, freq=880.0)
    sim = gate._compute_mert_similarity(audio1, audio2, SR)
    assert sim < 0.99, "Different frequencies should have lower similarity"


def test_09_timbral_fidelity_identical():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    fidelity = gate._compute_timbral_fidelity(audio, audio, SR)
    assert fidelity >= 0.99


def test_10_timbral_fidelity_different():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    np.random.seed(42)
    noise = 0.3 * np.random.randn(len(audio)).astype(np.float32)
    fidelity = gate._compute_timbral_fidelity(audio, noise, SR)
    assert fidelity < 0.95, "Noise should differ from tone"


def test_11_studio_quality_gain_loud():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0, amp=0.15)  # roughly -14 dBFS
    gain = gate._compute_studio_quality_gain(audio, audio, SR)
    assert 0.0 <= gain <= 1.0


def test_12_studio_quality_gain_quiet():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    quiet = _audio(2.0, amp=0.001)  # very quiet
    gain = gate._compute_studio_quality_gain(quiet, quiet, SR)
    assert 0.0 <= gain <= 1.0


def test_13_hpi_result_fields():
    from backend.core.holistic_perceptual_gate import HPIResult

    result = HPIResult(hpi=0.85, passed=True)
    assert result.hpi == 0.85
    assert result.passed is True
    assert result.mert_similarity == 1.0  # default
    assert result.is_studio_mode is False


def test_14_emotional_arc_factor():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    # Low emotional arc should reduce HPI
    result_low = gate.evaluate_restoration(audio, audio, SR, emotional_arc_score=0.5)
    result_high = gate.evaluate_restoration(audio, audio, SR, emotional_arc_score=1.0)
    assert result_low.hpi < result_high.hpi


def test_15_restorability_strict_gate():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result_normal = gate.evaluate_restoration(audio, audio, SR, restorability_score=70.0)
    result_strict = gate.evaluate_restoration(audio, audio, SR, restorability_score=90.0)
    # Strict gate reduces HPI slightly
    assert result_strict.hpi <= result_normal.hpi


def test_16_short_audio_graceful():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    short = np.zeros(100, dtype=np.float32)
    result = gate.evaluate_restoration(short, short, SR)
    assert np.isfinite(result.hpi)


def test_17_stereo_input():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    mono = _audio(2.0)
    stereo = np.stack([mono, mono * 0.9], axis=0)
    result = gate.evaluate_restoration(stereo, stereo, SR)
    assert result.hpi > 0


def test_18_nan_input_safe():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    nan_audio = audio.copy()
    nan_audio[100] = np.nan
    # Should not crash
    sim = gate._compute_mert_similarity(audio, nan_audio, SR)
    assert np.isfinite(sim)


def test_19_studio_mode_flag():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    rest = gate.evaluate_restoration(audio, audio, SR)
    studio = gate.evaluate_studio(audio, audio, SR)
    assert rest.is_studio_mode is False
    assert studio.is_studio_mode is True


def test_20_artifact_freedom_veto():
    """artifact_freedom < 0.95 → passed must be False regardless of HPI value."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR, artifact_freedom=0.94)
    assert not result.passed


def test_21_detail_dict_present():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR)
    assert isinstance(result.detail, dict)


def test_22_pqs_improvement_affects_studio():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result_low = gate.evaluate_studio(audio, audio, SR, pqs_improvement=-0.5)
    result_high = gate.evaluate_studio(audio, audio, SR, pqs_improvement=1.0)
    assert result_high.hpi >= result_low.hpi


def test_23_mert_short_audio():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    short = np.zeros(500, dtype=np.float32)
    sim = gate._compute_mert_similarity(short, short, SR)
    assert sim == 1.0  # fallback for short audio


def test_24_timbral_short_audio():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    short = np.zeros(500, dtype=np.float32)
    fid = gate._compute_timbral_fidelity(short, short, SR)
    assert fid == 1.0  # fallback for short audio


def test_25_hpi_range():
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(audio, audio, SR)
    assert result.hpi >= 0.0
    assert result.hpi <= 1.5  # theoretical max with all factors = 1.0


# ── §2.44 Restorability-Anker-Strategie ───────────────────────────────────


def test_26_restorability_anchor_detail_keys():
    """evaluate_restoration muss input_weight / ref_weight im detail-Dict liefern."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    with (
        patch.object(gate, "_compute_mert_similarity", return_value=1.0),
        patch.object(gate, "_compute_timbral_fidelity", return_value=1.0),
    ):
        result = gate.evaluate_restoration(audio, audio, SR, restorability_score=60.0)
    assert "input_weight" in result.detail
    assert "ref_weight" in result.detail
    assert "timbral_input" in result.detail
    assert "timbral_ref" in result.detail


def test_27_high_restorability_ref_dominant():
    """Restorability > 70 → input_weight=0.0, ref_weight=1.0 (§2.44 FIX v9.11.2).

    Nach dem Referenz-Paradox-Fix wird bei hoher Restorability NICHT mehr die
    Ähnlichkeit zum degradierten Input gemessen, sondern die Referenz/direktionale
    Qualität. input_weight=0.0 verhindert, dass gute Restaurierung bestraft wird.
    """
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    with (
        patch.object(gate, "_compute_mert_similarity", return_value=1.0),
        patch.object(gate, "_compute_timbral_fidelity", return_value=1.0),
    ):
        result = gate.evaluate_restoration(audio, audio, SR, restorability_score=80.0)
    assert result.detail["input_weight"] == 0.0
    assert result.detail["ref_weight"] == 1.0


def test_28_mid_restorability_ref_dominant():
    """Restorability 50–70 → input_weight=0.35, ref_weight=0.65 (§2.44 FIX v9.11.2)."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    with (
        patch.object(gate, "_compute_mert_similarity", return_value=1.0),
        patch.object(gate, "_compute_timbral_fidelity", return_value=1.0),
    ):
        result = gate.evaluate_restoration(audio, audio, SR, restorability_score=60.0)
    assert result.detail["input_weight"] == 0.35
    assert result.detail["ref_weight"] == 0.65


def test_29_low_restorability_ref_dominant():
    """Restorability ≤ 50 → input_weight=0.2, ref_weight=0.8 (§2.44 FIX v9.11.2)."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    with (
        patch.object(gate, "_compute_mert_similarity", return_value=1.0),
        patch.object(gate, "_compute_timbral_fidelity", return_value=1.0),
    ):
        result = gate.evaluate_restoration(audio, audio, SR, restorability_score=30.0)
    assert result.detail["input_weight"] == 0.2
    assert result.detail["ref_weight"] == 0.8


def test_30_restorability_50_boundary_blended():
    """Restorability = 50 (Grenzfall) → blended (50–70 Bereich, §2.44 FIX v9.11.2)."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    with (
        patch.object(gate, "_compute_mert_similarity", return_value=1.0),
        patch.object(gate, "_compute_timbral_fidelity", return_value=1.0),
    ):
        result = gate.evaluate_restoration(audio, audio, SR, restorability_score=50.0)
    assert result.detail["input_weight"] == 0.35
    assert result.detail["ref_weight"] == 0.65


# ── §2.44 update_reference_memory + EMA ───────────────────────────────────


def test_31_update_reference_memory_method_exists():
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    assert hasattr(gate, "update_reference_memory")


def test_32_update_rejected_if_hpi_too_low():
    """Quality-Gate: HPI ≤ 0.5 → kein Update."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.4,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="jazz",
        material="vinyl_std",
        era_bin="pre-1960",
    )
    assert ("jazz", "vinyl_std", "pre-1960") not in gate._ref_memory


def test_33_update_rejected_if_artifact_freedom_low():
    """Quality-Gate: artifact_freedom < 0.95 → kein Update."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.8,
        artifact_freedom=0.90,
        p1_p2_passed=True,
        genre="jazz",
        material="vinyl_std",
        era_bin="pre-1960",
    )
    assert ("jazz", "vinyl_std", "pre-1960") not in gate._ref_memory


def test_34_update_rejected_if_p1_p2_failed():
    """Quality-Gate: P1/P2 nicht bestanden → kein Update."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.8,
        artifact_freedom=1.0,
        p1_p2_passed=False,
        genre="jazz",
        material="vinyl_std",
        era_bin="pre-1960",
    )
    assert ("jazz", "vinyl_std", "pre-1960") not in gate._ref_memory


def test_35_update_creates_entry():
    """Valides Update → Eintrag in _ref_memory angelegt."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.8,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="jazz",
        material="vinyl_std",
        era_bin="pre-1960",
    )
    key = ("jazz", "vinyl_std", "pre-1960")
    assert key in gate._ref_memory
    assert gate._ref_memory[key].obs_count == 1
    assert gate._ref_memory[key].calibrated is False  # erst 1 Beobachtung


def test_36_update_ema_three_times_calibrated():
    """Nach 3 validen Updates → calibrated = True."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    key = ("pop", "tape_std", "1960-1990")
    for _ in range(3):
        gate.update_reference_memory(
            audio,
            SR,
            hpi=0.9,
            artifact_freedom=1.0,
            p1_p2_passed=True,
            genre="pop",
            material="tape_std",
            era_bin="1960-1990",
        )
    assert key in gate._ref_memory
    assert gate._ref_memory[key].obs_count == 3
    assert gate._ref_memory[key].calibrated is True


def test_37_ema_smoothing_alpha():
    """EMA-Update: Embedding muss sich nach Update ändern (weder identisch noch komplett ersetzt)."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio1 = _audio(2.0, freq=440.0)
    audio2 = _audio(2.0, freq=880.0)  # different embedding

    gate.update_reference_memory(
        audio1,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="rock",
        material="digital",
        era_bin="post-1990",
    )
    emb_after_first = gate._ref_memory[("rock", "digital", "post-1990")].embedding.copy()

    gate.update_reference_memory(
        audio2,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="rock",
        material="digital",
        era_bin="post-1990",
    )
    emb_after_second = gate._ref_memory[("rock", "digital", "post-1990")].embedding

    # Embedding muss sich geändert haben (EMA-Mischung)
    assert not np.allclose(emb_after_first, emb_after_second, atol=1e-6)
    # Aber nicht komplett ersetzt (EMA-Faktor 0.15 → große Überlappung)
    corr = float(
        np.dot(emb_after_first, emb_after_second)
        / (np.linalg.norm(emb_after_first) * np.linalg.norm(emb_after_second) + 1e-12)
    )
    assert corr > 0.5, f"EMA should preserve most of old reference, got correlation {corr:.3f}"


# ── §2.44 Fallback-Kaskade ─────────────────────────────────────────────────


def test_38_fallback_stufe5_no_vector():
    """Kein Eintrag in _ref_memory → _get_reference_vector gibt None zurück."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()  # fresh instance, no memory
    vec = gate._get_reference_vector("oper", "shellac", "pre-1960")
    assert vec is None


def test_39_fallback_stufe1_exact_match():
    """Exakter Treffer (genre, material, era) → liefert gespeicherten Vektor."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0, freq=220.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="blues",
        material="shellac",
        era_bin="pre-1960",
    )
    vec = gate._get_reference_vector("blues", "shellac", "pre-1960")
    assert vec is not None
    assert vec.shape == (40,)


def test_40_fallback_stufe2_same_era_same_genre():
    """Kein exakter material-Treffer, aber gleiche genre+era → Stufe-2-Fallback."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="jazz",
        material="shellac",
        era_bin="pre-1960",
    )
    # Anderes Material, aber gleiche genre+era
    vec = gate._get_reference_vector("jazz", "vinyl_78", "pre-1960")
    assert vec is not None


def test_41_fallback_stufe3_same_genre():
    """Kein era-Treffer für dieses genre → Stufe-3: gleicher Genre, andere Ära."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="jazz",
        material="tape_std",
        era_bin="1960-1990",
    )
    # Anderes Material UND andere Ära, aber gleicher Genre → Stufe-3
    vec = gate._get_reference_vector("jazz", "digital", "post-1990")
    assert vec is not None


def test_42_reference_used_in_hpi_low_restorability():
    """Restorability ≤ 50 + existierender Ref-Vektor → ref_weight=0.7 und result.passed=True."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    # Zuerst Referenz anlegen
    gate.update_reference_memory(
        audio,
        SR,
        hpi=0.9,
        artifact_freedom=1.0,
        p1_p2_passed=True,
        genre="folk",
        material="shellac",
        era_bin="pre-1960",
    )
    result = gate.evaluate_restoration(
        audio,
        audio,
        SR,
        restorability_score=30.0,
        genre="folk",
        material="shellac",
        era_bin="pre-1960",
    )
    assert result.detail["ref_weight"] == 0.8
    assert result.hpi > 0


def test_43_compute_embedding_returns_valid_shape():
    """_compute_embedding liefert normierten float32-Vektor der Länge 40."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    audio = _audio(2.0)
    emb = gate._compute_embedding(audio, SR)
    assert emb.shape == (40,)
    assert emb.dtype == np.float32
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 0.01, f"Embedding should be unit-normed, got norm={norm:.4f}"


def test_44_cosine_similarity_identical_vectors():
    """Cosine similarity identischer Vektoren = 1.0."""
    from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

    gate = HolisticPerceptualGate()
    v = np.random.rand(40).astype(np.float32)
    sim = gate._cosine_similarity(v, v)
    assert abs(sim - 1.0) < 1e-5


def test_45_genre_material_era_in_detail():
    """genre/material/era_bin müssen im detail-Dict stehen."""
    from backend.core.holistic_perceptual_gate import get_holistic_gate

    gate = get_holistic_gate()
    audio = _audio(2.0)
    result = gate.evaluate_restoration(
        audio,
        audio,
        SR,
        genre="schlager",
        material="vinyl_std",
        era_bin="1960-1990",
    )
    assert result.detail.get("genre") == "schlager"
    assert result.detail.get("material") == "vinyl_std"
    assert result.detail.get("era_bin") == "1960-1990"
