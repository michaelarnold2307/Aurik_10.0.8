import unittest

import numpy as np
from restoration_workflow import RestorationWorkflow


class TestRestorationWorkflow(unittest.TestCase):
    def setUp(self):
        self.workflow = RestorationWorkflow()
        self.feedback_events = []
        self.workflow.feedback_bus.subscribe(lambda event, data: self.feedback_events.append((event, data)))

    def test_full_workflow_declicking(self):
        image_path = "covers/vinyl_rock.jpg"
        prompt = "Mehr Brillanz"
        audio_meta = {"material": "vinyl"}
        original = np.ones(1000)
        processed = np.ones(1000) * 0.8
        result = self.workflow.process(image_path, prompt, audio_meta, original, processed, "declicking")
        self.assertIn("decision", result)
        self.assertIn("metrics", result)
        self.assertIn("explanation", result)
        self.assertTrue(any(e[0] == "metrics" for e in self.feedback_events))
        self.assertTrue(any(e[0] == "explanation" for e in self.feedback_events))

    def test_full_workflow_eq(self):
        image_path = "covers/unknown.jpg"
        prompt = "wärmer"
        audio_meta = {"material": "digital"}
        original = np.ones(1000)
        processed = np.ones(1000) * 1.1
        result = self.workflow.process(image_path, prompt, audio_meta, original, processed, "eq")
        self.assertIn("spectral_balance", result["metrics"])
        self.assertIn("explanation", result)


if __name__ == "__main__":
    unittest.main()
