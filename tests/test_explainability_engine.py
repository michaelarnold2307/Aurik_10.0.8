import unittest

from explainability_engine import ExplainabilityEngine


class TestExplainabilityEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ExplainabilityEngine()

    def test_declicking_explanation(self):
        msg = self.engine.explain("declicking", {}, {"click_reduction": 0.7})
        self.assertIn("viele Störimpulse", msg)
        msg2 = self.engine.explain("declicking", {}, {"click_reduction": 0.1})
        self.assertIn("kaum nötig", msg2)

    def test_denoising_explanation(self):
        msg = self.engine.explain("denoising", {}, {"noise_reduction": 0.5})
        self.assertIn("hoher Rauschpegel", msg)
        msg2 = self.engine.explain("denoising", {}, {"noise_reduction": 0.1})
        self.assertIn("minimal", msg2)

    def test_eq_explanation(self):
        msg = self.engine.explain("eq", {}, {"bark_band_deviation": 0.2})
        self.assertIn("spektrale Unausgewogenheiten", msg)
        msg2 = self.engine.explain("eq", {}, {"bark_band_deviation": 0.01})
        self.assertIn("kaum nötig", msg2)

    def test_default_explanation(self):
        msg = self.engine.explain("unknown", {}, {})
        self.assertIn("Phase", msg)


if __name__ == "__main__":
    unittest.main()
