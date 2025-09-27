"""Public accessors for filesystem locations used by Property Adviser."""
from __future__ import annotations

from property_adviser.core.paths import (
    CONFIG_DIR,
    DATA_DIR,
    FEATURE_ENGINEERING_CONFIG_PATH,
    MODEL_CONFIG_PATH,
    MODELS_DIR,
    PREPROCESS_CONFIG_PATH,
    PREPROCESS_DIR,
    PROJECT_ROOT,
    STREET_COORDS_PATH,
    TRAINING_DIR,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CONFIG_DIR",
    "PREPROCESS_DIR",
    "TRAINING_DIR",
    "MODELS_DIR",
    "PREPROCESS_CONFIG_PATH",
    "FEATURE_ENGINEERING_CONFIG_PATH",
    "MODEL_CONFIG_PATH",
    "STREET_COORDS_PATH",
]
