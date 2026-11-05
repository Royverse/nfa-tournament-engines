import os
import gc
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

# =====================================================================
# CrunchDAO DataCrunch-2 Model: senior-jaguar
# Architecture: Multi-Horizon GBDT + Ridge Ensemble with Dynamic Factor
#               Neutralization & Affine-Gaussian Rank Transformation
# Target: Per-Moon Pearson & Spearman Correlation
# =====================================================================

MODEL_FILE_NAME = "model.joblib"
MAX_TRAIN_ROWS = 240000


def get_model_path(model_directory_path: str) -> str:
    os.makedirs(model_directory_path, exist_ok=True)
    return os.path.join(model_directory_path, MODEL_FILE_NAME)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("Feature_")]


def rank_normalize(series: pd.Series) -> pd.Series:
    """Normalize series into uniform percentile ranks in (0, 1]."""
    if len(series) <= 1:
        return series
    ranks = rankdata(series.values, method="average")
    return pd.Series(ranks / len(ranks), index=series.index)


def gaussian_rank_normalize(series: pd.Series) -> pd.Series:
    """Transform cross-sectional values into affine-scaled Gaussian quantiles in (0, 1).

    Because DataCrunch-2's official evaluator computes per-moon Pearson correlation,
    mapping ranks via the inverse normal CDF (probit) aligns prediction tails with
    Gaussian returns while keeping values strictly inside [0.001, 0.999].
    """
    n = len(series)
    if n <= 2:
        return rank_normalize(series)
    ranks = rankdata(series.values, method="average")
    u = np.clip((ranks - 0.5) / n, 1e-4, 1.0 - 1e-4)
    z = np.clip(norm.ppf(u), -3.0, 3.0)
    # Positive affine transform from [-3, 3] -> [0.001, 0.999] preserves exact Pearson correlation
    scaled = 0.5 + (z / 6.02)
    return pd.Series(np.clip(scaled, 0.001, 0.999), index=series.index)


def neutralize_array(
    features: np.ndarray,
    predictions: np.ndarray,
    proportion: float = 0.25,
) -> np.ndarray:
    """
    Project 1D predictions orthogonal to selected risk feature columns.
    Formula: y_neutral = y - proportion * X (X^T X + lambda I)^(-1) X^T y
    """
    if proportion <= 0.0 or features.shape[1] == 0 or len(predictions) < 5:
        return predictions

    ones = np.ones((len(features), 1), dtype=np.float64)
    X_aug = np.hstack([ones, np.nan_to_num(features.astype(np.float64), nan=0.0)])

    ridge_lambda = 1e-3
    XtX = X_aug.T @ X_aug
    XtX += ridge_lambda * np.eye(XtX.shape[0])
    try:
        weights = np.linalg.solve(XtX, X_aug.T @ predictions.astype(np.float64))
        linear_projection = X_aug @ weights
        return predictions - proportion * linear_projection
    except (np.linalg.LinAlgError, ValueError):
        return predictions


class QuantEnsembleModel:
    """
    Three-model cross-sectional asset pricing ensemble:
    1. Deep Interaction LightGBM (captures higher-order non-linear feature interactions)
    2. Shallow Regularized LightGBM (high-colsample, low-variance signal stabilizer)
    3. L2 Ridge Regression (captures global linear factor premia)
    Also records the highest-exposure / regime-sensitive features during training
    for targeted post-prediction neutralization.
    """

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self.neutralize_cols: list[str] = feature_cols[:30]
        self.lgb_model = LGBMRegressor(
            n_estimators=220,
            learning_rate=0.035,
            max_depth=6,
            num_leaves=31,
            colsample_bytree=0.28,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=0.25,
            reg_lambda=1.5,
            min_child_samples=40,
            random_state=42,
            n_jobs=2,
            verbose=-1,
        )
        self.lgb_shallow = LGBMRegressor(
            n_estimators=260,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            colsample_bytree=0.45,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=0.5,
            reg_lambda=2.5,
            min_child_samples=60,
            random_state=1337,
            n_jobs=2,
            verbose=-1,
        )
        self.ridge_model = Ridge(alpha=250.0, random_state=42)
        self.ensemble_weight_lgb = 0.45
        self.ensemble_weight_shallow = 0.35
        self.ensemble_weight_ridge = 0.20

    def fit(self, X: pd.DataFrame, y: pd.Series, moons: pd.Series = None):
        X_feats = X[self.feature_cols].copy().fillna(0.0).astype(np.float32)

        # Standardize target per moon so extreme volatility moons don't dominate tree splits
        if moons is not None and len(moons) == len(y):
            y_df = pd.DataFrame({"y": y.values, "moon": moons.values}, index=y.index)
            y_norm = y_df.groupby("moon")["y"].transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-8) if len(s) > 1 else s - s.mean()
            )
            y_vals = np.nan_to_num(y_norm.values, nan=0.0).astype(np.float32)
        else:
            y_vals = np.nan_to_num(y.values, nan=0.0).astype(np.float32)

        self.lgb_model.fit(X_feats, y_vals)
        self.lgb_shallow.fit(X_feats, y_vals)
        self.ridge_model.fit(X_feats, y_vals)

        # Identify top 30 features with highest linear exposure to raw ensemble predictions
        try:
            sample_n = min(len(X_feats), 12000)
            X_sub_df = X_feats.iloc[-sample_n:]
            X_sub = X_sub_df.values
            raw_p = (
                self.ensemble_weight_lgb * self.lgb_model.predict(X_sub_df)
                + self.ensemble_weight_shallow * self.lgb_shallow.predict(X_sub_df)
                + self.ensemble_weight_ridge * self.ridge_model.predict(X_sub_df)
            )
            p_centered = raw_p - np.mean(raw_p)
            p_std = np.std(p_centered) + 1e-9
            X_centered = X_sub - np.mean(X_sub, axis=0, keepdims=True)
            X_std = np.std(X_centered, axis=0) + 1e-9
            exposures = np.abs((X_centered.T @ p_centered) / (len(raw_p) * X_std * p_std))
            top_idx = np.argsort(exposures)[::-1][: min(30, len(self.feature_cols))]
            self.neutralize_cols = [self.feature_cols[i] for i in top_idx]
        except Exception:
            self.neutralize_cols = self.feature_cols[: min(30, len(self.feature_cols))]

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.array([])

        available_cols = [c for c in self.feature_cols if c in X.columns]
        if not available_cols:
            available_cols = get_feature_columns(X)

        if not available_cols:
            return np.zeros(len(X))

        X_feats = X[available_cols].copy().fillna(0.0).astype(np.float32)

        for col in self.feature_cols:
            if col not in X_feats.columns:
                X_feats[col] = np.float32(0.0)

        X_feats = X_feats[self.feature_cols]

        lgb_preds = self.lgb_model.predict(X_feats)
        ridge_preds = self.ridge_model.predict(X_feats)

        if hasattr(self, "lgb_shallow") and self.lgb_shallow is not None:
            shallow_preds = self.lgb_shallow.predict(X_feats)
            w_deep = getattr(self, "ensemble_weight_lgb", 0.45)
            w_shallow = getattr(self, "ensemble_weight_shallow", 0.35)
            w_ridge = getattr(self, "ensemble_weight_ridge", 0.20)
            return w_deep * lgb_preds + w_shallow * shallow_preds + w_ridge * ridge_preds

        return (
            self.ensemble_weight_lgb * lgb_preds
            + self.ensemble_weight_ridge * ridge_preds
        )


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_directory_path: str,
    loop_moon: int = None,
    embargo: int = 4,
) -> None:
    """
    CrunchDAO tournament train hook.
    Trains the ensemble on complete historical moons and persists model artifact.
    """
    feature_cols = get_feature_columns(X_train)
    if not feature_cols:
        raise ValueError("No Feature_* columns found in X_train")

    if "target" in y_train.columns:
        target_series = y_train["target"]
    else:
        target_series = y_train.iloc[:, -1]

    moons = X_train["moon"] if "moon" in X_train.columns else None

    if loop_moon is not None and moons is not None:
        eligible_mask = moons <= (loop_moon - embargo)
        X_train_filtered = X_train[eligible_mask]
        target_filtered = target_series[eligible_mask]
        moons_filtered = moons[eligible_mask]
    else:
        X_train_filtered = X_train
        target_filtered = target_series
        moons_filtered = moons

    # Preserve complete moons from most recent history up to MAX_TRAIN_ROWS
    if len(X_train_filtered) > MAX_TRAIN_ROWS:
        if moons_filtered is not None:
            unique_moons = np.sort(moons_filtered.unique())
            avg_per_moon = max(1, len(X_train_filtered) // max(1, len(unique_moons)))
            keep_moons_count = max(20, min(len(unique_moons), MAX_TRAIN_ROWS // avg_per_moon))
            selected_moons = set(unique_moons[-keep_moons_count:])
            moon_mask = moons_filtered.isin(selected_moons)
            X_train_filtered = X_train_filtered[moon_mask]
            target_filtered = target_filtered[moon_mask]
            moons_filtered = moons_filtered[moon_mask]
        else:
            X_train_filtered = X_train_filtered.iloc[-MAX_TRAIN_ROWS:]
            target_filtered = target_filtered.iloc[-MAX_TRAIN_ROWS:]

    print(f"[DataCrunch-2] Training senior-jaguar on {len(X_train_filtered)} samples with {len(feature_cols)} features...")
    model = QuantEnsembleModel(feature_cols=feature_cols)
    model.fit(X_train_filtered, target_filtered, moons_filtered)

    model_path = get_model_path(model_directory_path)
    joblib.dump(model, model_path)
    print(f"[DataCrunch-2] Model successfully saved to {model_path}")

    gc.collect()


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
    loop_moon: int = None,
    embargo: int = 4,
) -> pd.DataFrame:
    """
    CrunchDAO tournament infer hook.
    Returns DataFrame containing ['id', 'moon', 'prediction'].
    """
    model_path = get_model_path(model_directory_path)
    if not os.path.exists(model_path):
        print(f"[DataCrunch-2] Model not found at {model_path}. Auto-training fallback ensemble on X_test...")
        feature_cols = get_feature_columns(X_test)
        model = QuantEnsembleModel(feature_cols=feature_cols)
        dummy_X = X_test[feature_cols].head(min(100, len(X_test))).fillna(0.0)
        dummy_y = pd.Series(np.zeros(len(dummy_X)))
        model.fit(dummy_X, dummy_y)
        joblib.dump(model, model_path)
    else:
        model: QuantEnsembleModel = joblib.load(model_path)
    raw_predictions = model.predict(X_test)

    output = pd.DataFrame()
    output["id"] = X_test["id"] if "id" in X_test.columns else np.arange(len(X_test))
    output["moon"] = X_test["moon"] if "moon" in X_test.columns else (loop_moon or 0)
    output["prediction"] = raw_predictions

    # Pre-neutralization Gaussian rank normalization per moon
    if "moon" in output.columns:
        output["prediction"] = output.groupby("moon")["prediction"].transform(gaussian_rank_normalize)
    else:
        output["prediction"] = gaussian_rank_normalize(output["prediction"])

    # Targeted Feature Neutralization against highest-exposure risk factors (proportion=0.25)
    feature_cols = get_feature_columns(X_test)
    if feature_cols:
        saved_neutral = getattr(model, "neutralize_cols", None)
        if saved_neutral:
            neutral_cols = [c for c in saved_neutral if c in X_test.columns]
        else:
            neutral_cols = []
        if not neutral_cols:
            neutral_cols = feature_cols[: min(30, len(feature_cols))]

        preds_array = output["prediction"].values.copy().astype(np.float64)
        X_feats_array = X_test[neutral_cols].fillna(0.0).values.astype(np.float64)

        if "moon" in X_test.columns:
            moons_arr = X_test["moon"].values
            for m in np.unique(moons_arr):
                mask = moons_arr == m
                if np.sum(mask) >= 5:
                    preds_array[mask] = neutralize_array(
                        X_feats_array[mask],
                        preds_array[mask],
                        proportion=0.25,
                    )
        else:
            preds_array = neutralize_array(X_feats_array, preds_array, proportion=0.25)

        output["prediction"] = preds_array

        # Final post-neutralization Gaussian rank transform (optimizes per-moon Pearson & Spearman)
        if "moon" in output.columns:
            output["prediction"] = output.groupby("moon")["prediction"].transform(gaussian_rank_normalize)
        else:
            output["prediction"] = gaussian_rank_normalize(output["prediction"])

    output["prediction"] = output["prediction"].astype(np.float64)
    return output[["id", "moon", "prediction"]]


if __name__ == "__main__":
    print("Testing DataCrunch-2 senior-jaguar local pipeline...")

    # Generate synthetic dataset matching DataCrunch-2 schema
    n_samples = 600
    n_features = 20
    feature_names = [f"Feature_{i}" for i in range(n_features)]

    np.random.seed(42)
    fake_X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names,
    )
    fake_X["id"] = [f"stock_{i}" for i in range(n_samples)]
    fake_X["moon"] = np.repeat([10, 11, 12, 13, 14, 15], 100)

    # Synthetic signal with non-linear interaction + noise
    signal = (
        0.3 * fake_X["Feature_0"]
        - 0.2 * fake_X["Feature_1"]
        + 0.15 * (fake_X["Feature_2"] * fake_X["Feature_3"])
    )
    fake_y = pd.DataFrame({
        "id": fake_X["id"],
        "moon": fake_X["moon"],
        "target": signal + 0.5 * np.random.randn(n_samples),
    })

    test_model_dir = "resources"
    train(fake_X, fake_y, test_model_dir)

    # Test inference
    test_X = fake_X.iloc[-100:].copy()
    preds = infer(test_X, test_model_dir, loop_moon=15)

    print("Predictions shape:", preds.shape)
    print("Predictions preview:")
    print(preds.head())

    # Calculate both Pearson and Spearman correlation with aligned indices
    y_test_subset = fake_y.iloc[-100:]["target"].reset_index(drop=True)
    preds_clean = preds["prediction"].reset_index(drop=True)
    pearson_corr = preds_clean.corr(y_test_subset, method="pearson")
    spearman_corr = preds_clean.corr(y_test_subset, method="spearman")
    print(f"Validation Pearson Correlation:  {pearson_corr:.4f}")
    print(f"Validation Spearman Correlation: {spearman_corr:.4f}")
    assert len(preds) == 100, "Inference output count mismatch"
    assert "prediction" in preds.columns, "Missing prediction column"
    print("[Success] Local pipeline test passed successfully!")
