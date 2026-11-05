"""Unit tests for the NFA CrunchDAO Tournament Engine."""
import os
import sys
import unittest

# Ensure crunchdao directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfa_crunchdao.config import DEFAULT_COMPETITION_NAME, DEFAULT_MODEL_NAME
from nfa_crunchdao.client import get_auth_token, verify_credentials
from nfa_crunchdao.model import get_feature_columns, calculate_spearman_score

class TestCrunchDAOConfig(unittest.TestCase):
    def test_default_config(self):
        self.assertEqual(DEFAULT_COMPETITION_NAME, "datacrunch-2")
        self.assertEqual(DEFAULT_MODEL_NAME, "royverse")

    def test_auth_token_parsing(self):
        # 1. Test CRUNCHDAO_TOKEN
        os.environ["CRUNCHDAO_TOKEN"] = "token_abc123456789"
        self.assertEqual(get_auth_token(), "token_abc123456789")
        creds = verify_credentials()
        self.assertTrue(creds["valid"])
        # No fragment of the token may come back: Actions only masks the whole secret.
        self.assertNotIn("token_preview", creds)
        self.assertNotIn("6789", repr(creds))

        # 2. Test Bearer prefix stripping
        os.environ["CRUNCHDAO_TOKEN"] = "Bearer token_xyz987654321"
        self.assertEqual(get_auth_token(), "token_xyz987654321")

        # 3. Clean up
        os.environ.pop("CRUNCHDAO_TOKEN", None)
        self.assertIsNone(get_auth_token())
        invalid_creds = verify_credentials()
        self.assertFalse(invalid_creds["valid"])

    def test_feature_column_filtering(self):
        # Mock simple dataframe-like dictionary columns
        class MockDF:
            columns = ["id", "moon", "date", "target_b", "target_g", "feature_1", "feature_2", "volatility"]
            def select_dtypes(self, include=None):
                return self
        mock_df = MockDF()
        features = get_feature_columns(mock_df)
        self.assertIn("feature_1", features)
        self.assertIn("feature_2", features)
        self.assertIn("volatility", features)
        self.assertNotIn("id", features)
        self.assertNotIn("moon", features)
        self.assertNotIn("target_b", features)
        self.assertNotIn("target_g", features)

    def test_spearman_score_calculation(self):
        try:
            import numpy as np
            y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
            score = calculate_spearman_score(y_true, y_pred)
            self.assertAlmostEqual(score, 1.0, places=4)

            # Test inverted order
            y_inv = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
            score_inv = calculate_spearman_score(y_true, y_inv)
            self.assertAlmostEqual(score_inv, -1.0, places=4)
        except ImportError:
            self.skipTest("numpy or scipy not installed in test runner environment")

if __name__ == "__main__":
    unittest.main()
