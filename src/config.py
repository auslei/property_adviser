from pathlib import Path

# --- Directories ---
DATA_DIR = Path("data")
PREPROCESS_DIR = Path("data_preprocess")
TRAINING_DIR = Path("data_training")
MODELS_DIR = Path("models")
CONFIG_DIR = Path("config")

# --- File Paths ---
STREET_COORDS_PATH = CONFIG_DIR / "street_coordinates.csv"
PREPROCESS_CONFIG_PATH = CONFIG_DIR / "preprocessing.yml"
FEATURE_ENGINEERING_CONFIG_PATH = CONFIG_DIR / "features.yml"
MODEL_CONFIG_PATH = CONFIG_DIR / "model.yml"

# --- Patterns and Constants ---
RAW_DATA_PATTERN = "*.csv"
RANDOM_STATE = 42
MIN_NON_NULL_FRACTION = 0.35
