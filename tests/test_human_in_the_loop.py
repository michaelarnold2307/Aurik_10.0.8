"""
Integrationstest für Human-in-the-Loop: Experten- und Community-Feedback, Aggregation, Integration.
"""

import unittest

from community_rating_platform import CommunityRatingPlatform
from expert_feedback_system import ExpertFeedbackSystem
from feedback_integrator import FeedbackIntegrator


class TestHumanInTheLoop(unittest.TestCase):
    def test_expert_feedback(self):
        system = ExpertFeedbackSystem()
        system.add_feedback("Expert1", {"brillanz": 0.9, "waerme": 0.8})
        system.add_feedback("Expert2", {"brillanz": 0.8, "waerme": 0.9})
        agg = system.aggregate()
        self.assertAlmostEqual(agg["brillanz"], 0.85)
        self.assertAlmostEqual(agg["waerme"], 0.85)

    def test_community_rating(self):
        platform = CommunityRatingPlatform()
        platform.add_rating("User1", {"brillanz": 0.7, "waerme": 0.8})
        platform.add_rating("User2", {"brillanz": 0.8, "waerme": 0.7})
        agg = platform.aggregate()
        self.assertAlmostEqual(agg["brillanz"], 0.75)
        self.assertAlmostEqual(agg["waerme"], 0.75)

    def test_feedback_integration(self):
        system = ExpertFeedbackSystem()
        platform = CommunityRatingPlatform()
        integrator = FeedbackIntegrator(system, platform)
        system.add_feedback("Expert", {"brillanz": 0.9, "waerme": 0.8})
        platform.add_rating("User", {"brillanz": 0.7, "waerme": 0.6})
        integrated = integrator.integrate()
        self.assertAlmostEqual(integrated["brillanz"], 0.8)
        self.assertAlmostEqual(integrated["waerme"], 0.7)


if __name__ == "__main__":
    unittest.main()
