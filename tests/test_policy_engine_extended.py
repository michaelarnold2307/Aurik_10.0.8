import logging

import numpy as np

from policy.policy_engine import PolicyEngine


def test_policy_engine_extended_quality_gates():
    # Dummy-Audio und Testdaten
    audio = np.zeros(48000)
    sr = 48000
    media_characteristics = {
        "vocal": True,
        "chain": True,
        "music": True,
        "creative_dsp_chain": ["HarmonicExciter", "StereoWidener"],
    }
    vocal_scores = {"authentizität": 0.89, "klarheit": 0.91, "expressivität": 0.88}
    media_history = {"original_chain": ["RMSEnergy", "ZeroCrossingRate", "HarmonicExciter", "StereoWidener"]}
    feedback_data = {"expert": "Sehr natürlich", "user": "Klang exzellent"}
    policy = {"goal": "music_enhancement"}
    engine = PolicyEngine(policy)
    result = engine.process(
        audio,
        sr,
        user_score=5.0,
        user_comment="Sehr musikalisch",
        media_characteristics=media_characteristics,
        vocal_scores=vocal_scores,
        media_history=media_history,
        feedback_data=feedback_data,
    )
    # Prüfe Quality-Gates und Audit-Log
    assert result["quality_passed"] is True
    assert "vocal_quality" in result
    assert "chain_authenticity" in result
    assert "feedback_optimizer" in result
    assert "regression_monitor" in result
    assert "release_check" in result
    assert "user_feedback_analyzer" in result
    import logging

    logging.info("Erweiterte Quality-Gates und Policy-Engine Test bestanden.")


def test_policy_engine_chain_authenticity_negative():
    # Dummy-Audio und Testdaten
    audio = np.zeros(48000)
    sr = 48000
    media_characteristics = {
        "vocal": True,
        "chain": True,
        "music": True,
        "creative_dsp_chain": ["HarmonicExciter", "StereoWidener"],
    }
    vocal_scores = {"authentizität": 0.89, "klarheit": 0.91, "expressivität": 0.88}
    # media_history mit abweichender Kette
    media_history = {"original_chain": ["Limiter", "StereoWidener"]}
    feedback_data = {"expert": "Sehr natürlich", "user": "Klang exzellent"}
    policy = {"goal": "music_enhancement"}
    engine = PolicyEngine(policy)
    result = engine.process(
        audio,
        sr,
        user_score=5.0,
        user_comment="Sehr musikalisch",
        media_characteristics=media_characteristics,
        vocal_scores=vocal_scores,
        media_history=media_history,
        feedback_data=feedback_data,
    )
    # Die Kette stimmt nicht überein, daher muss authentic False sein
    assert "chain_authenticity" in result
    assert result["chain_authenticity"].get("authentic") is False
    logging.info("Negativer ChainAuthenticity-Test bestanden.")
