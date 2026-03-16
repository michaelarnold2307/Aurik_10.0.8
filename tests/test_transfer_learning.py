"""
Integrationstest für Transfer Learning: Cross-domain adaptation (Vinyl → Tape, etc.)
"""

import unittest

import numpy as np
from transfer_learner import TransferLearner


class TestTransferLearner(unittest.TestCase):
    def test_transfer_learning(self):
        # Simulierte Daten: Vinyl (Quelle), Tape (Ziel)
        X_source = np.random.rand(100, 10)
        y_source = np.random.rand(100)
        X_target = np.random.rand(20, 10)
        # Ziel: Transfer von Vinyl-Modell auf Tape-Daten
        learner = TransferLearner(source_domain="vinyl", target_domain="tape")
        learner.fit(X_source, y_source)
        y_pred = learner.transfer(X_target)
        self.assertEqual(y_pred.shape[0], X_target.shape[0])
        self.assertTrue(np.all(np.isfinite(y_pred)))


if __name__ == "__main__":
    unittest.main()
