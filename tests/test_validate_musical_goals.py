import numpy as np
from validate_musical_goals import (
    ArtifactChecker,
    FormantGuard,
    MixBalanceChecker,
    PitchContourChecker,
    VoiceMatchChecker,
)


def test_voice_match_checker():
    checker = VoiceMatchChecker()
    orig = np.array([0.1, 0.2, 0.3])
    proc = np.array([0.1, 0.2, 0.3])
    assert checker.check(orig, proc)
    proc2 = np.array([0.9, 0.8, 0.7])
    assert not checker.check(orig, proc2)


def test_formant_guard():
    checker = FormantGuard()
    orig = {"f1_mean": 500, "f2_mean": 1500}
    proc = {"f1_mean": 505, "f2_mean": 1510}
    assert checker.check(orig, proc)
    proc2 = {"f1_mean": 530, "f2_mean": 1550}
    assert not checker.check(orig, proc2)


def test_mix_balance_checker():
    checker = MixBalanceChecker()
    orig = {"vocals": 0, "bass": 0, "drums": 0, "other": 0}
    proc = {"vocals": 0.5, "bass": 0.5, "drums": 0.5, "other": 0.5}
    assert not checker.check(orig, proc)
    proc2 = {"vocals": 0.8, "bass": 0.8, "drums": 0.8, "other": 0.8}
    assert not checker.check(orig, proc2)
    proc3 = {"vocals": 0.5, "bass": 0.5, "drums": 0.5, "other": 0.5}
    assert not checker.check(orig, proc3)


def test_pitch_contour_checker():
    checker = PitchContourChecker()
    orig = np.array([1, 2, 3, 4])
    proc = np.array([1, 2, 3, 4])
    assert checker.check(orig, proc)
    proc2 = np.array([4, 3, 2, 1])
    assert not checker.check(orig, proc2)


def test_artifact_checker_klang_aesthetik():
    checker = ArtifactChecker.KlangAesthetikChecker()
    scores = {
        "brillanz": 0.8,
        "wärme": 0.8,
        "natürlichkeit": 0.8,
        "authentizität": 0.8,
        "emotionalität": 0.8,
        "transparenz": 0.8,
    }
    assert checker.check(scores)
    scores2 = {
        "brillanz": 0.6,
        "wärme": 0.6,
        "natürlichkeit": 0.6,
        "authentizität": 0.6,
        "emotionalität": 0.6,
        "transparenz": 0.6,
    }
    assert not checker.check(scores2)


def test_artifact_checker_exzellenz():
    checker = ArtifactChecker.ExzellenzChecker()
    assert checker.check(0.5, 1.3)
    assert not checker.check(0.5, 1.0)
