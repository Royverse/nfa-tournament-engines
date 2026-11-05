"""Configuration for the NFA Numerai Tournament Engine.

Tournament 8 is Numerai Classic (crowdsourced global equity predictions).
Model: 'royverse'
"""
import os

TOURNAMENT_ID = 8  # Classic
DEFAULT_MODEL_NAME = "royverse"
DEFAULT_MODEL_ID = "43d6a18d-95e4-4c98-9e3d-713ec5f67cf1"

DATA_VERSION = "v5.3"
TARGET_NAME = "target"
ENSEMBLE_TARGETS = [
    "target_ender_60",
    "target_ender_20",
    "target_jasper_60",
    "target_factor_feat_neutral_60",
    "target_eleven_60",
    "target_jupiter_20",
    "target",
]
ENSEMBLE_TARGET_WEIGHTS = {
    "target_ender_60": 0.35,
    "target_ender_20": 0.30,
    "target_jasper_60": 0.20,
    "target_factor_feat_neutral_60": 0.35,
    "target_eleven_60": 0.30,
    "target_jupiter_20": 0.20,
    "target": 0.15,
}
DOWNSAMPLE_EVERY = 3
FEATURE_SET = "medium"  # Balanced performance / RAM / speed

NUM_TREES = 1500
LEARNING_RATE = 0.01
MAX_DEPTH = 6
NUM_LEAVES = 64
COLSAMPLE_BYTREE = 0.15
NEUTRALIZE_PROPORTION = 0.30
NEUTRALIZE_TOP_K = 40
BENCHMARK_ORTHOGONALIZE_PROPORTION = 0.25
SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("NUMERAI_DATA_DIR", os.path.join(BASE_DIR, "data"))
MODELS_DIR = os.getenv("NUMERAI_MODELS_DIR", os.path.join(BASE_DIR, "models"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
