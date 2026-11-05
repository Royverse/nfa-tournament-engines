"""CrunchDAO Main Entry Point.

This file provides the official standard train() and infer() interface required by
CrunchDAO's competition runner and crunch-cli.
"""
import os
import sys
import pandas as pd
from typing import Any

# Ensure local package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nfa_crunchdao.model import (
    train_model as _train_model,
    infer_predictions as _infer_predictions,
    load_or_train_model,
)
from nfa_crunchdao.config import RESOURCES_DIR

def train(
    X_train: pd.DataFrame,
    y_train: Any,
    model_directory_path: str = RESOURCES_DIR,
):
    """Train the model and save artifacts into model_directory_path."""
    print(f"[CrunchDAO] Running train() with {len(X_train):,} training samples...")
    booster = _train_model(X_train, y_train, model_directory_path=model_directory_path)
    print(f"[CrunchDAO] train() completed successfully.")
    return booster

def infer(
    X_test: pd.DataFrame,
    model_directory_path: str = RESOURCES_DIR,
) -> pd.DataFrame:
    """Generate inferences using pre-trained model saved in model_directory_path."""
    print(f"[CrunchDAO] Running infer() with {len(X_test):,} inference samples...")
    model = load_or_train_model(model_directory_path=model_directory_path)
    sub_df = _infer_predictions(model, X_test, model_directory_path=model_directory_path)
    print(f"[CrunchDAO] infer() generated {len(sub_df):,} predictions.")
    return sub_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CrunchDAO Standard Runner")
    parser.add_argument("--test", action="store_true", help="Run local crunch-cli test")
    args = parser.parse_args()

    if args.test:
        try:
            import crunch
            crunch_tools = crunch.load_notebook()
            crunch_tools.test()
        except ImportError:
            print("crunch library not installed. Run: pip install crunch-cli")
