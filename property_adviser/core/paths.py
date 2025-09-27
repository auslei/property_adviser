"""Centralised filesystem locations for Property Adviser artefacts.

These constants are deliberately lightweight (no UI-specific imports) so they can be
consumed by CLI tools, tests, and notebooks without pulling in heavier dependencies.
Paths are resolved relative to the repository root.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESS_DIR = PROJECT_ROOT / "data" / "preprocess"
TRAINING_DIR = PROJECT_ROOT / "data" / "training"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
PREPROCESS_CONFIG_PATH = CONFIG_DIR / "preprocessing.yml"
FEATURE_ENGINEERING_CONFIG_PATH = CONFIG_DIR / "features.yml"
MODEL_CONFIG_PATH = CONFIG_DIR / "model.yml"
STREET_COORDS_PATH = CONFIG_DIR / "street_coordinates.csv"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "PREPROCESS_DIR",
    "TRAINING_DIR",
    "MODELS_DIR",
    "CONFIG_DIR",
    "PREPROCESS_CONFIG_PATH",
    "FEATURE_ENGINEERING_CONFIG_PATH",
    "MODEL_CONFIG_PATH",
    "STREET_COORDS_PATH",
]
