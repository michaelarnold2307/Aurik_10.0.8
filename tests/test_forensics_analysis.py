"""
Test suite for forensics/analysis_and_modules.py
Tests PolicyManager and FeatureExtractor classes
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.forensics.analysis_and_modules import FeatureExtractor, PolicyManager


class TestPolicyManager:

    def test_policy_manager_init(self):
        """Test PolicyManager initialization"""
        policy = {"threshold": 0.5}
        pm = PolicyManager(policy)

        assert pm.policy == {"threshold": 0.5}
        assert pm.escalation_levels["warn"] == 3
        assert pm.escalation_levels["bypass"] == 5
        assert pm.escalation_levels["hard_bypass"] == 7

    def test_policy_manager_custom_escalation(self):
        """Test PolicyManager with custom escalation levels"""
        policy = {}
        escalation = {"warn": 2, "bypass": 4, "hard_bypass": 6}
        pm = PolicyManager(policy, escalation_levels=escalation)

        assert pm.escalation_levels["warn"] == 2
        assert pm.escalation_levels["bypass"] == 4

    def test_policy_update_fail_count(self):
        """Test policy update increments fail count"""
        policy = {}
        pm = PolicyManager(policy)

        # Trigger failure
        feedback = {"quality_gate_1": False}
        pm.update(feedback)

        assert "quality_gate_1" in pm.policy
        assert pm.policy["quality_gate_1"]["fail_count"] == 1
        assert pm.policy["quality_gate_1"]["escalated"] == False

    def test_policy_escalation_warn_level(self):
        """Test policy escalation to warn level"""
        policy = {}
        pm = PolicyManager(policy)

        # Trigger 3 failures to reach warn level
        for _ in range(3):
            pm.update({"gate_test": False})

        assert pm.policy["gate_test"]["fail_count"] == 3
        assert pm.policy["gate_test"]["escalated"] == True
        assert pm.policy["gate_test"]["escalation_level"] == "warn"
        assert pm.policy["gate_test"]["action"] == "warn"

    def test_policy_escalation_bypass_level(self):
        """Test policy escalation to bypass level"""
        policy = {}
        pm = PolicyManager(policy)

        # Trigger 5 failures to reach bypass level
        for _ in range(5):
            pm.update({"gate_test": False})

        assert pm.policy["gate_test"]["fail_count"] == 5
        assert pm.policy["gate_test"]["escalation_level"] == "bypass"
        assert pm.policy["gate_test"]["action"] == "bypass_or_notify"

    def test_policy_escalation_hard_bypass(self):
        """Test policy escalation to hard_bypass level"""
        policy = {}
        pm = PolicyManager(policy)

        # Trigger 7 failures to reach hard_bypass level
        for _ in range(7):
            pm.update({"gate_test": False})

        assert pm.policy["gate_test"]["fail_count"] == 7
        assert pm.policy["gate_test"]["escalation_level"] == "hard_bypass"
        assert pm.policy["gate_test"]["action"] == "hard_bypass"

    def test_policy_success_no_increment(self):
        """Test that successful gates don't increment fail count"""
        policy = {}
        pm = PolicyManager(policy)

        # First failure
        pm.update({"gate_test": False})
        assert pm.policy["gate_test"]["fail_count"] == 1

        # Erfolg setzt fail_count zurück
        pm.update({"gate_test": True})
        assert pm.policy["gate_test"]["fail_count"] == 0

    def test_policy_callback_invoked(self):
        """Test that callback is invoked on escalation"""
        policy = {}
        callback_events = []

        def test_callback(event):
            callback_events.append(event)

        pm = PolicyManager(policy, callback=test_callback)

        # Trigger escalation
        for _ in range(3):
            pm.update({"gate_test": False})

        # Callback should have been invoked
        assert len(callback_events) > 0
        assert callback_events[0]["event"] == "escalation"
        assert callback_events[0]["gate"] == "gate_test"
        assert callback_events[0]["level"] == "warn"

    def test_policy_log_created(self):
        """Test that policy log is created"""
        policy = {}
        pm = PolicyManager(policy)

        # Trigger escalation to create log entry
        for _ in range(3):
            pm.update({"gate_test": False})

        assert "_log" in pm.policy
        assert len(pm.policy["_log"]) > 0
        # Robust: Mindestens ein Log-Eintrag mit event=="escalation"
        assert any(entry.get("event") == "escalation" for entry in pm.policy["_log"])


class TestFeatureExtractor:
    def test_feature_extractor_init(self):
        """Test FeatureExtractor initialization"""
        fe = FeatureExtractor()
        assert fe is not None

    def test_extract_basic(self):
        """Test basic feature extraction"""
        fe = FeatureExtractor()

        # Create simple test audio
        sr = 48000
        duration = 1.0
        audio = np.random.randn(int(sr * duration)) * 0.5

        # Create simple policy manager
        policy_manager = PolicyManager({})

        # Extract features
        features = fe.extract(audio, sr, reference=None, policy_manager=policy_manager)

        # Check that features is a dict
        assert isinstance(features, dict)

    def test_extract_with_reference(self):
        """Test feature extraction with reference audio"""
        fe = FeatureExtractor()

        sr = 48000
        duration = 1.0
        audio = np.random.randn(int(sr * duration)) * 0.5
        reference = np.random.randn(int(sr * duration)) * 0.5

        policy_manager = PolicyManager({})

        features = fe.extract(audio, sr, reference=reference, policy_manager=policy_manager)

        assert isinstance(features, dict)

    def test_extract_stereo(self):
        """Test feature extraction with stereo audio"""
        fe = FeatureExtractor()

        sr = 48000
        duration = 0.5
        # Stereo audio (2 channels)
        audio = np.random.randn(2, int(sr * duration)) * 0.5

        policy_manager = PolicyManager({})

        # Should handle stereo gracefully
        features = fe.extract(audio, sr, reference=None, policy_manager=policy_manager)

        assert isinstance(features, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
