"""Model training, loading, inference, and prediction validation."""
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from .config import (
    BENCHMARK_ORTHOGONALIZE_PROPORTION,
    COLSAMPLE_BYTREE,
    DATA_DIR,
    DATA_VERSION,
    DOWNSAMPLE_EVERY,
    ENSEMBLE_TARGETS,
    ENSEMBLE_TARGET_WEIGHTS,
    FEATURE_SET,
    LEARNING_RATE,
    MAX_DEPTH,
    MODELS_DIR,
    NEUTRALIZE_PROPORTION,
    NEUTRALIZE_TOP_K,
    NUM_LEAVES,
    NUM_TREES,
    SEED,
    TARGET_NAME,
)

MODEL_FILE = os.path.join(MODELS_DIR, "lgb_baseline.txt")
ENSEMBLE_META_FILE = os.path.join(MODELS_DIR, "lgb_ensemble_meta.json")


class NumeraiMultiTargetEnsemble:
    """Wrapper holding multiple target-specific LightGBM boosters with a unified .predict() API."""

    def __init__(self, boosters: Dict[str, lgb.Booster], weights: Dict[str, float]):
        self.boosters = boosters
        self.weights = weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.boosters:
            return np.zeros(len(X), dtype=np.float64)
        if len(self.boosters) == 1:
            only_booster = next(iter(self.boosters.values()))
            return np.asarray(only_booster.predict(X), dtype=np.float64)

        blended = np.zeros(len(X), dtype=np.float64)
        total_w = 0.0
        for target_col, booster in self.boosters.items():
            w = float(self.weights.get(target_col, 1.0))
            raw = np.asarray(booster.predict(X), dtype=np.float64)
            # Standardize each horizon/target model before blending so 60d and 20d scales align
            std = float(np.std(raw))
            normed = (raw - float(np.mean(raw))) / (std if std > 1e-9 else 1.0)
            blended += w * normed
            total_w += w
        return blended / max(total_w, 1e-9)


def get_feature_names(features_json_path: str, set_name: str = FEATURE_SET) -> List[str]:
    """Extract list of feature column names from features.json."""
    with open(features_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    sets = meta.get("feature_sets", {})
    if set_name in sets:
        return sets[set_name]
    if "medium" in sets:
        return sets["medium"]
    if "small" in sets:
        return sets["small"]
    return sets.get("all", [])


def _available_parquet_columns(parquet_path: str) -> Optional[set]:
    """Inspect parquet schema without loading data into memory."""
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(parquet_path)
        return set(schema.names)
    except Exception:
        return None


def neutralize_predictions(
    predictions: np.ndarray,
    exposures_matrix: np.ndarray,
    proportion: float = NEUTRALIZE_PROPORTION,
) -> np.ndarray:
    """Project predictions orthogonal to exposure columns via ridge-stabilized OLS."""
    if proportion <= 0.0 or exposures_matrix.shape[1] == 0 or len(predictions) < 10:
        return predictions

    X_clean = np.nan_to_num(exposures_matrix.astype(np.float64), nan=0.0)
    X_centered = X_clean - np.mean(X_clean, axis=0, keepdims=True)
    ones = np.ones((len(predictions), 1), dtype=np.float64)
    X_aug = np.hstack([ones, X_centered])

    XtX = X_aug.T @ X_aug + 1e-3 * np.eye(X_aug.shape[1])
    try:
        weights = np.linalg.solve(XtX, X_aug.T @ predictions.astype(np.float64))
        projection = X_aug @ weights
        return predictions - proportion * projection
    except Exception:
        return predictions


def train_or_load_model(train_parquet_path: str, features_json_path: str) -> Any:
    """Load existing multi-target ensemble (or baseline booster), or train a new multi-target ensemble."""
    if os.path.exists(ENSEMBLE_META_FILE):
        try:
            with open(ENSEMBLE_META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            boosters: Dict[str, lgb.Booster] = {}
            for target_col, rel_file in meta.get("files", {}).items():
                full_path = os.path.join(MODELS_DIR, rel_file)
                if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                    boosters[target_col] = lgb.Booster(model_file=full_path)
            if boosters:
                print(f"[Model] Loaded multi-target ensemble ({list(boosters.keys())}) from {MODELS_DIR}")
                return NumeraiMultiTargetEnsemble(boosters, meta.get("weights", ENSEMBLE_TARGET_WEIGHTS))
        except Exception as e:
            print(f"[Model] Warning loading ensemble metadata: {e}")

    if os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 0:
        print(f"[Model] Loading existing booster from {MODEL_FILE}")
        return lgb.Booster(model_file=MODEL_FILE)

    print(f"[Model] No saved booster found. Training new multi-target LightGBM ensemble...")
    feature_cols = get_feature_names(features_json_path)

    schema_cols = _available_parquet_columns(train_parquet_path)
    if schema_cols is not None:
        target_cols = [t for t in ENSEMBLE_TARGETS if t in schema_cols]
        if not target_cols and TARGET_NAME in schema_cols:
            target_cols = [TARGET_NAME]
        feature_cols = [c for c in feature_cols if c in schema_cols]
    else:
        target_cols = [TARGET_NAME]

    if not target_cols:
        target_cols = [TARGET_NAME]

    cols_to_load = list(dict.fromkeys(["era"] + target_cols + feature_cols))
    print(
        f"[Model] Loading columns from {train_parquet_path} "
        f"({len(feature_cols)} features, targets={target_cols})..."
    )
    train_df = pd.read_parquet(train_parquet_path, columns=cols_to_load)

    # Downsample eras with recency preservation (always include the most recent era)
    eras = train_df["era"].unique()
    sampled_eras = list(eras[::DOWNSAMPLE_EVERY])
    if len(eras) > 0 and eras[-1] not in sampled_eras:
        sampled_eras.append(eras[-1])
    train_df = train_df[train_df["era"].isin(set(sampled_eras))].dropna(subset=[target_cols[0]])
    print(f"[Model] Sampled {len(train_df):,} rows across {len(sampled_eras)} eras")

    X_train = train_df[feature_cols].values

    boosters: Dict[str, lgb.Booster] = {}
    saved_files: Dict[str, str] = {}

    for idx, target_col in enumerate(target_cols):
        y_raw = train_df[target_col].fillna(0.5).values
        params = {
            "objective": "regression",
            "metric": "mse",
            "learning_rate": LEARNING_RATE,
            "num_leaves": NUM_LEAVES,
            "max_depth": MAX_DEPTH,
            "feature_fraction": COLSAMPLE_BYTREE,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_child_samples": 100,
            "lambda_l1": 0.2,
            "lambda_l2": 2.0,
            "verbosity": -1,
            "n_jobs": -1,
            "seed": SEED + idx * 17,
        }

        ds = lgb.Dataset(X_train, label=y_raw, feature_name=feature_cols, free_raw_data=False)
        t0 = time.time()
        booster = lgb.train(params, ds, num_boost_round=NUM_TREES)
        print(f"[Model] Trained {target_col} booster in {time.time() - t0:.1f}s")

        target_filename = f"lgb_{target_col}.txt"
        target_filepath = os.path.join(MODELS_DIR, target_filename)
        booster.save_model(target_filepath)
        boosters[target_col] = booster
        saved_files[target_col] = target_filename

        # Also save primary booster to MODEL_FILE for backward compatibility
        if idx == 0 or target_col == TARGET_NAME:
            booster.save_model(MODEL_FILE)

    with open(ENSEMBLE_META_FILE, "w", encoding="utf-8") as f:
        json.dump({"files": saved_files, "weights": ENSEMBLE_TARGET_WEIGHTS}, f, indent=2)

    print(f"[Model] Multi-target ensemble saved to {MODELS_DIR} (primary -> {MODEL_FILE})")
    return NumeraiMultiTargetEnsemble(boosters, ENSEMBLE_TARGET_WEIGHTS)


def train_live_distilled_ensemble(
    live_parquet_path: str,
    features_json_path: str,
    live_benchmark_parquet_path: Optional[str] = None,
    live_example_parquet_path: Optional[str] = None,
) -> NumeraiMultiTargetEnsemble:
    """Train an in-memory multi-target distilled LightGBM ensemble on live features + benchmark targets.

    Used during Numerai's Saturday round-opening window when `v5.3/train.parquet` is
    temporarily off-line for weekly era appending and no cached model exists yet.
    Does not write to MODELS_DIR so the next weekday run caches the full historical model.
    """
    print("[Model] Building in-memory multi-target distilled ensemble from live benchmark models...")
    live_df = pd.read_parquet(live_parquet_path)
    feature_cols = [c for c in get_feature_names(features_json_path) if c in live_df.columns]
    X_live = live_df[feature_cols].fillna(0.5).values

    teacher_targets: Dict[str, np.ndarray] = {}
    teacher_weights: Dict[str, float] = {}

    if live_benchmark_parquet_path and os.path.exists(live_benchmark_parquet_path):
        bm_df = pd.read_parquet(live_benchmark_parquet_path).reindex(live_df.index)
        if "v53_lgbm_ender60" in bm_df.columns:
            teacher_targets["v53_lgbm_ender60"] = bm_df["v53_lgbm_ender60"].fillna(0.5).values
            teacher_weights["v53_lgbm_ender60"] = 0.50
        if "v53_lgbm_ender20" in bm_df.columns:
            teacher_targets["v53_lgbm_ender20"] = bm_df["v53_lgbm_ender20"].fillna(0.5).values
            teacher_weights["v53_lgbm_ender20"] = 0.30
        if not teacher_targets:
            num_cols = [c for c in bm_df.select_dtypes(include=[np.number]).columns if c != "era"]
            for c in num_cols[:3]:
                teacher_targets[c] = bm_df[c].fillna(0.5).values
                teacher_weights[c] = 1.0 / max(1, len(num_cols[:3]))

    if live_example_parquet_path and os.path.exists(live_example_parquet_path):
        ex_df = pd.read_parquet(live_example_parquet_path).reindex(live_df.index)
        if "prediction" in ex_df.columns:
            teacher_targets["example_prediction"] = ex_df["prediction"].fillna(0.5).values
            teacher_weights["example_prediction"] = 0.20

    if not teacher_targets:
        raise RuntimeError("No benchmark or example targets available for live distillation fallback.")

    boosters: Dict[str, lgb.Booster] = {}
    for idx, (t_name, y_teacher) in enumerate(teacher_targets.items()):
        params = {
            "objective": "regression",
            "metric": "mse",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 5,
            "feature_fraction": 0.25,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_child_samples": 40,
            "lambda_l1": 0.5,
            "lambda_l2": 3.0,
            "verbosity": -1,
            "n_jobs": -1,
            "seed": SEED + idx * 31,
        }
        ds = lgb.Dataset(X_live, label=y_teacher, feature_name=feature_cols, free_raw_data=False)
        boosters[t_name] = lgb.train(params, ds, num_boost_round=350)
        print(f"[Model] Distilled student booster for {t_name} ({len(X_live):,} live rows)")

    return NumeraiMultiTargetEnsemble(boosters, teacher_weights)


def generate_predictions(
    booster: Any,
    live_parquet_path: str,
    features_json_path: str,
    live_benchmark_parquet_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Generate neutralized, benchmark-orthogonalized ranked predictions for the live round."""
    print(f"[Model] Reading live data from {live_parquet_path}...")
    live_df = pd.read_parquet(live_parquet_path)
    feature_cols = get_feature_names(features_json_path)

    # Impute missing feature columns with neutral 2 (if int8 binned 0..4) or 0.5
    for c in feature_cols:
        if c not in live_df.columns:
            live_df[c] = 0.5

    X_live = live_df[feature_cols].fillna(0.5).values
    raw_preds = np.asarray(booster.predict(X_live), dtype=np.float64)

    # 1. Dynamic Feature Neutralization against top-K highest exposure features
    if NEUTRALIZE_PROPORTION > 0.0 and X_live.shape[1] > 0 and len(raw_preds) >= 20:
        X_float = X_live.astype(np.float64)
        p_c = raw_preds - np.mean(raw_preds)
        p_s = np.std(p_c) + 1e-9
        X_c = X_float - np.mean(X_float, axis=0, keepdims=True)
        X_s = np.std(X_c, axis=0) + 1e-9
        exposures = np.abs((X_c.T @ p_c) / (len(raw_preds) * X_s * p_s))
        top_k = min(NEUTRALIZE_TOP_K, X_float.shape[1])
        top_idx = np.argsort(exposures)[::-1][:top_k]
        raw_preds = neutralize_predictions(
            raw_preds,
            X_float[:, top_idx],
            proportion=NEUTRALIZE_PROPORTION,
        )
        print(f"[Model] Neutralized predictions against top-{top_k} risk features (proportion={NEUTRALIZE_PROPORTION})")

    # 2. Optional Live Benchmark Orthogonalization (boosts BMC & MMC)
    if (
        live_benchmark_parquet_path
        and os.path.exists(live_benchmark_parquet_path)
        and BENCHMARK_ORTHOGONALIZE_PROPORTION > 0.0
    ):
        try:
            bm_df = pd.read_parquet(live_benchmark_parquet_path)
            bm_aligned = bm_df.reindex(live_df.index)
            bm_cols = [
                c for c in bm_aligned.select_dtypes(include=[np.number]).columns
                if c not in {"era", "data_type"}
            ]
            if bm_cols:
                bm_mat = bm_aligned[bm_cols[:8]].fillna(0.5).values.astype(np.float64)
                raw_preds = neutralize_predictions(
                    raw_preds,
                    bm_mat,
                    proportion=BENCHMARK_ORTHOGONALIZE_PROPORTION,
                )
                print(
                    f"[Model] Orthogonalized against {min(8, len(bm_cols))} live benchmark models "
                    f"(proportion={BENCHMARK_ORTHOGONALIZE_PROPORTION})"
                )
        except Exception as e:
            print(f"[Model] Benchmark orthogonalization skipped: {e}")

    # 3. Rank normalize into [0, 1] as required by Numerai
    ranked_preds = pd.Series(raw_preds, index=live_df.index).rank(pct=True).values

    # Validation checks
    assert len(ranked_preds) == len(live_df), "Prediction row count mismatch"
    assert not np.isnan(ranked_preds).any(), "NaN found in predictions"
    assert not np.isinf(ranked_preds).any(), "Inf found in predictions"
    assert ranked_preds.min() >= 0.0 and ranked_preds.max() <= 1.0, "Predictions outside [0, 1]"

    sub_df = pd.DataFrame({"prediction": ranked_preds}, index=live_df.index)
    output_path = os.path.join(DATA_DIR, "submission.csv")
    sub_df.to_csv(output_path)

    print(f"[Model] Generated {len(sub_df):,} predictions. Stats: min={ranked_preds.min():.4f}, "
          f"mean={ranked_preds.mean():.4f}, max={ranked_preds.max():.4f}")
    return sub_df, output_path
