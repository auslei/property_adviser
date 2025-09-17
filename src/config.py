import json
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path("data")
PREPROCESS_DIR = Path("data_preprocess")
TRAINING_DIR = Path("data_training")
MODELS_DIR = Path("models")
CONFIG_DIR = Path("config")
USER_CONFIG_PATH = CONFIG_DIR / "settings.json"

RAW_DATA_PATTERN = "*.csv"
RANDOM_STATE = 42
MIN_NON_NULL_FRACTION = 0.35


def _load_user_config() -> Dict[str, object]:
    if not USER_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(USER_CONFIG_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {USER_CONFIG_PATH}: {exc}"
        ) from exc


USER_CONFIG = _load_user_config()

# Columns listed here will be removed before feature engineering / modelling.
EXCLUDE_COLUMNS: List[str] = [
    col for col in USER_CONFIG.get("exclude_columns", []) if isinstance(col, str)
]
