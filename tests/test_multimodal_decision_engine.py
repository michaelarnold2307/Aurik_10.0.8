import unittest

from multimodal_decision_engine import MultimodalDecisionEngine


class TestMultimodalDecisionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MultimodalDecisionEngine()

    def test_decision_with_vinyl_cover_and_prompt(self):
        image_path = "covers/vinyl_rock.jpg"
        prompt = "Mehr Brillanz und weniger Rauschen"
        audio_meta = {"material": "vinyl"}
        decision = self.engine.decide(image_path, prompt, audio_meta)
        self.assertIn("brilliance_enhancer", decision["chain"])
        self.assertIn("denoiser", decision["chain"])
        self.assertEqual(decision["meta"]["era"], "1970s")

    def test_decision_with_jazz_cover(self):
        image_path = "covers/jazz_album.jpg"
        prompt = ""
        audio_meta = {"material": "vinyl"}
        decision = self.engine.decide(image_path, prompt, audio_meta)
        self.assertIn("warmth_enhancer", decision["chain"])
        self.assertEqual(decision["meta"]["genre"], "Jazz")

    def test_decision_with_prompt_only(self):
        image_path = "covers/unknown.jpg"
        prompt = "wärmer"
        audio_meta = {"material": "digital"}
        decision = self.engine.decide(image_path, prompt, audio_meta)
        self.assertIn("warmth_enhancer", decision["chain"])
        self.assertEqual(decision["meta"]["eq_low"], 1.1)


if __name__ == "__main__":
    unittest.main()
