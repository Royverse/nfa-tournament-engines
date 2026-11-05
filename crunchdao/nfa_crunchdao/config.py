"""Configuration for the NFA CrunchDAO Tournament Engine.

Competes in CrunchDAO equity & statistical forecasting tournaments (DataCrunch / Synth).
Zero-capital model slot: 'royverse'
Payouts: USDC direct rewards.
"""
import os

DEFAULT_COMPETITION_NAME = os.getenv("CRUNCHDAO_COMPETITION_NAME", os.getenv("CRUNCH_COMPETITION_NAME", "datacrunch-2"))
DEFAULT_MODEL_NAME = os.getenv("CRUNCHDAO_MODEL_NAME", os.getenv("CRUNCH_MODEL_NAME", "royverse"))
DEFAULT_TARGET_NAME = os.getenv("CRUNCHDAO_TARGET", "target_b")

# Hyperparameters for LightGBM rank-preserving predictor
NUM_TREES = 1000
LEARNING_RATE = 0.02
MAX_DEPTH = 6
NUM_LEAVES = 64
SEED = 42
DOWNSAMPLE_EVERY = 4  # Era/Moon downsampling for fast memory-safe training

# Paths and storage
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("CRUNCHDAO_DATA_DIR", os.path.join(BASE_DIR, "data"))
RESOURCES_DIR = os.getenv("CRUNCHDAO_RESOURCES_DIR", os.path.join(BASE_DIR, "resources"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESOURCES_DIR, exist_ok=True)
