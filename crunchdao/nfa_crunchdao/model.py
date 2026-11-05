"""CrunchDAO Quantitative Model Training, Inference, and Validation Engine."""
import os
import sys
import time
from typing import List, Tuple, Optional, Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

from .config import (
    DEFAULT_TARGET_NAME,
    DOWNSAMPLE_EVERY,
    LEARNING_RATE,
    MAX_DEPTH,
    NUM_LEAVES,
    NUM_TREES,
    RESOURCES_DIR,
    SEED,
)

MODEL_FILENAME = "lgb_crunchdao_model.joblib"

def get_feature_columns(df: Any) -> List[str]:
    """Identify quantitative feature columns from dataframe."""
    excluded = {"id", "moon", "era", "date", "target", "target_b", "target_g", "target_r", "target_w"}
    feature_cols = [c for c in df.columns if c not in excluded and not str(c).startswith("target")]
    if not feature_cols and hasattr(df, "select_dtypes") and np is not None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    return feature_cols

def calculate_spearman_score(y_true: Any, y_pred: Any) -> float:
    """Calculate Spearman rank correlation between predictions and targets."""
    if spearmanr is None or np is None:
        raise ImportError("scipy and numpy are required to compute Spearman correlation.")
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0

def train_model(
    X_train: Any,
    y_train: Any,
    model_directory_path: str = RESOURCES_DIR,
) -> Any:
    """Train a LightGBM regressor on CrunchDAO training data and save to model_directory_path."""
    if lgb is None or joblib is None or np is None or pd is None:
        raise ImportError("lightgbm, joblib, numpy, and pandas are required for model training.")

    os.makedirs(model_directory_path, exist_ok=True)
    model_path = os.path.join(model_directory_path, MODEL_FILENAME)

    feature_cols = get_feature_columns(X_train)
    print(f"[CrunchDAO Model] Detected {len(feature_cols)} features for training")

    # Format y_train to 1D array
    if isinstance(y_train, pd.DataFrame):
        y_val = y_train.iloc[:, 0].values
    elif isinstance(y_train, pd.Series):
        y_val = y_train.values
    else:
        y_val = np.array(y_train).ravel()

    # Downsample if moon/era column is present to conserve memory & speed
    X_clean = X_train[feature_cols].copy()
    if "moon" in X_train.columns:
        moons = X_train["moon"].unique()
        sampled = moons[::DOWNSAMPLE_EVERY]
        mask = X_train["moon"].isin(sampled)
        X_mat = X_clean[mask].values
        y_mat = y_val[mask]
        print(f"[CrunchDAO Model] Downsampled to {len(X_mat):,} rows across {len(sampled)} moons")
    else:
        X_mat = X_clean.values
        y_mat = y_val

    # Impute missing values with 0.0
    if np.isnan(X_mat).any():
        print("[CrunchDAO Model] Imputing missing feature values with 0.0")
        X_mat = np.nan_to_num(X_mat, nan=0.0)

    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": LEARNING_RATE,
        "num_leaves": NUM_LEAVES,
        "max_depth": MAX_DEPTH,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 80,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "n_jobs": -1,
        "seed": SEED,
    }

    ds = lgb.Dataset(X_mat, label=y_mat, feature_name=feature_cols, free_raw_data=False)
    t0 = time.time()
    booster = lgb.train(params, ds, num_boost_round=NUM_TREES)
    print(f"[CrunchDAO Model] Training complete in {time.time() - t0:.1f}s")

    # Save booster artifact
    joblib.dump(booster, model_path)
    print(f"[CrunchDAO Model] Saved model to {model_path}")
    return booster

def load_or_train_model(
    X_train: Optional[Any] = None,
    y_train: Optional[Any] = None,
    model_directory_path: str = RESOURCES_DIR,
) -> Any:
    """Load existing booster if present, otherwise train a new one."""
    if joblib is None:
        raise ImportError("joblib is required to load models.")
    model_path = os.path.join(model_directory_path, MODEL_FILENAME)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f"[CrunchDAO Model] Loading cached model from {model_path}")
        return joblib.load(model_path)

    if X_train is not None and y_train is not None:
        return train_model(X_train, y_train, model_directory_path=model_directory_path)

    raise FileNotFoundError(f"Model file not found at {model_path} and no training data provided.")

def _neutralize_matrix(features: Any, predictions: Any, proportion: float = 0.25) -> Any:
    """Project predictions orthogonal to top-exposure feature columns."""
    if proportion <= 0.0 or features.shape[1] == 0 or len(predictions) < 5:
        return predictions
    ones = np.ones((len(features), 1), dtype=np.float64)
    X_aug = np.hstack([ones, np.nan_to_num(features.astype(np.float64), nan=0.0)])
    XtX = X_aug.T @ X_aug + 1e-3 * np.eye(X_aug.shape[1])
    try:
        weights = np.linalg.solve(XtX, X_aug.T @ predictions.astype(np.float64))
        return predictions - proportion * (X_aug @ weights)
    except Exception:
        return predictions


def _affine_gaussian_rank(series: Any) -> Any:
    """Map values to affine-scaled Gaussian quantiles in [0.001, 0.999] (or uniform rank fallback)."""
    n = len(series)
    if n <= 2:
        return series.rank(pct=True)
    try:
        from scipy.stats import norm, rankdata
        ranks = rankdata(series.values, method="average")
        u = np.clip((ranks - 0.5) / n, 1e-4, 1.0 - 1e-4)
        z = np.clip(norm.ppf(u), -3.0, 3.0)
        return pd.Series(np.clip(0.5 + (z / 6.02), 0.001, 0.999), index=series.index)
    except Exception:
        return series.rank(pct=True)


def infer_predictions(
    model: Any,
    X_test: Any,
    model_directory_path: str = RESOURCES_DIR,
) -> Any:
    """Generate neutralized, affine-Gaussian rank-normalized predictions in [0, 1]."""
    if np is None or pd is None:
        raise ImportError("numpy and pandas are required for inference.")

    if model is None:
        model = load_or_train_model(model_directory_path=model_directory_path)

    feature_cols = get_feature_columns(X_test)
    X_mat = X_test[feature_cols].copy().values

    if np.isnan(X_mat).any():
        X_mat = np.nan_to_num(X_mat, nan=0.0)

    # Predict raw regression score
    raw_preds = np.asarray(model.predict(X_mat), dtype=np.float64)

    # Dynamic feature neutralization against top 25 highest-exposure features
    if X_mat.shape[1] > 0 and len(raw_preds) >= 10:
        p_c = raw_preds - np.mean(raw_preds)
        p_s = np.std(p_c) + 1e-9
        X_c = X_mat - np.mean(X_mat, axis=0, keepdims=True)
        X_s = np.std(X_c, axis=0) + 1e-9
        exposures = np.abs((X_c.T @ p_c) / (len(raw_preds) * X_s * p_s))
        top_idx = np.argsort(exposures)[::-1][: min(25, X_mat.shape[1])]
        X_top = X_mat[:, top_idx]
        if "moon" in X_test.columns:
            moons_arr = X_test["moon"].values
            for m in np.unique(moons_arr):
                mask = moons_arr == m
                if np.sum(mask) >= 5:
                    raw_preds[mask] = _neutralize_matrix(X_top[mask], raw_preds[mask], proportion=0.25)
        else:
            raw_preds = _neutralize_matrix(X_top, raw_preds, proportion=0.25)

    # Convert to affine-Gaussian percentiles [0.001, 0.999] per moon or globally
    if "moon" in X_test.columns:
        pred_series = pd.Series(raw_preds, index=X_test.index)
        ranked = pred_series.groupby(X_test["moon"]).transform(_affine_gaussian_rank).values
    else:
        ranked = _affine_gaussian_rank(pd.Series(raw_preds, index=X_test.index)).values

    # Strict validation checks
    assert len(ranked) == len(X_test), "Prediction row count mismatch"
    assert not np.isnan(ranked).any(), "NaN found in predictions"
    assert not np.isinf(ranked).any(), "Inf found in predictions"
    assert ranked.min() >= 0.0 and ranked.max() <= 1.0, "Predictions outside [0, 1] interval"

    sub_df = pd.DataFrame({"prediction": ranked}, index=X_test.index)
    return sub_df
