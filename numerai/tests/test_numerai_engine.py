import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from nfa_numerai.config import TOURNAMENT_ID, DEFAULT_MODEL_NAME, DATA_VERSION, ENSEMBLE_TARGETS
from nfa_numerai.client import get_auth_credentials
from nfa_numerai.model import get_feature_names, neutralize_predictions, NumeraiMultiTargetEnsemble

class TestNumeraiConfig(unittest.TestCase):
    def test_tournament_id(self):
        self.assertEqual(TOURNAMENT_ID, 8)
        self.assertEqual(DEFAULT_MODEL_NAME, "royverse")
        self.assertEqual(DATA_VERSION, "v5.3")
        self.assertIn("target_ender_60", ENSEMBLE_TARGETS)
        self.assertIn("target_ender_20", ENSEMBLE_TARGETS)

    def test_auth_credentials_parsing(self):
        # Test MCP format
        os.environ["NUMERAI_MCP_AUTH"] = "Token mock_pub$mock_sec"
        pub, sec = get_auth_credentials()
        self.assertEqual(pub, "mock_pub")
        self.assertEqual(sec, "mock_sec")

        # Test individual env vars override
        os.environ["NUMERAI_PUBLIC_ID"] = "override_pub"
        os.environ["NUMERAI_SECRET_KEY"] = "override_sec"
        pub2, sec2 = get_auth_credentials()
        self.assertEqual(pub2, "override_pub")
        self.assertEqual(sec2, "override_sec")

        # Clean up
        os.environ.pop("NUMERAI_MCP_AUTH", None)
        os.environ.pop("NUMERAI_PUBLIC_ID", None)
        os.environ.pop("NUMERAI_SECRET_KEY", None)

    def test_prediction_ranking(self):
        # Simulate raw model outputs
        raw_scores = np.array([0.12, -0.45, 0.88, 0.05, -0.10, 0.33])
        series = pd.Series(raw_scores)
        ranked = series.rank(pct=True).values

        # Check bounds
        self.assertTrue((ranked > 0.0).all())
        self.assertTrue((ranked <= 1.0).all())
        self.assertFalse(np.isnan(ranked).any())
        self.assertEqual(len(ranked), len(raw_scores))
        # Ensure order is preserved
        self.assertEqual(np.argmax(raw_scores), np.argmax(ranked))
        self.assertEqual(np.argmin(raw_scores), np.argmin(ranked))

    def test_feature_neutralization_reduces_exposure(self):
        rng = np.random.RandomState(42)
        risk_factor = rng.randn(50)
        alpha_signal = rng.randn(50)
        raw_preds = 0.8 * risk_factor + 0.2 * alpha_signal
        exposures = risk_factor.reshape(-1, 1)

        neutralized = neutralize_predictions(raw_preds, exposures, proportion=1.0)
        corr_before = abs(np.corrcoef(raw_preds, risk_factor)[0, 1])
        corr_after = abs(np.corrcoef(neutralized, risk_factor)[0, 1])
        self.assertLess(corr_after, 0.05)
        self.assertLess(corr_after, corr_before)

    def test_multi_target_ensemble_blending(self):
        class DummyBooster:
            def __init__(self, vec):
                self.vec = vec
            def predict(self, X):
                return self.vec

        b1 = DummyBooster(np.array([1.0, 2.0, 3.0, 4.0]))
        b2 = DummyBooster(np.array([10.0, 20.0, 30.0, 40.0]))
        ens = NumeraiMultiTargetEnsemble(
            {"target_ender_60": b1, "target_ender_20": b2},
            {"target_ender_60": 0.6, "target_ender_20": 0.4},
        )
        preds = ens.predict(np.zeros((4, 2)))
        self.assertEqual(len(preds), 4)
        self.assertTrue(np.all(np.diff(preds) > 0))

if __name__ == "__main__":
    unittest.main()
