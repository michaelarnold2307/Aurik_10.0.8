"""
Integrationstest für Psychoacoustic Optimization
"""

import unittest

import numpy as np

from backend.core.fletcher_munson_curves import FletcherMunsonProcessor
from backend.core.masking_analyzer import MaskingAnalyzer
from backend.core.psychoacoustic_core import BarkScaleProcessor


class TestPsychoacousticOptimization(unittest.TestCase):
    def test_bark_scale(self):
        processor = BarkScaleProcessor()
        spectrum = processor.analyze([0.1] * 48000, 48000)
        # BarkSpectrum besitzt vermutlich .bands oder .values
        if hasattr(spectrum, "bands"):
            self.assertEqual(len(spectrum.bands), 24)
        elif hasattr(spectrum, "values"):
            self.assertEqual(len(spectrum.values), 24)
        else:
            self.fail("BarkSpectrum besitzt keine bands/values für Längenprüfung")

    def test_masking(self):
        analyzer = MaskingAnalyzer()
        masked = analyzer.apply_masking([0.1] * 48000, 48000)
        self.assertEqual(len(masked), 48000)

    def test_equal_loudness(self):
        processor = FletcherMunsonProcessor()
        loudness, _ = processor.apply_compensation(np.array([0.1] * 48000), 48000)
        self.assertEqual(len(loudness), 48000)


if __name__ == "__main__":
    unittest.main()
