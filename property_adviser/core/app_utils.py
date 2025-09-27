import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from property_adviser.core.config import load_config
from property_adviser.core.paths import (
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
from property_adviser.core.io import load_parquet_or_csv

# Ensure repository root is importable for legacy notebooks or interactive shells.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def load_model_resources() -> Tuple[Dict[str, Any], object, Dict[str, Any]]:
    metadata_path = TRAINING_DIR / "feature_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Feature metadata missing. Run preprocessing and feature selection first."
        )
    metadata = json.loads(metadata_path.read_text())

    model_candidates = [
        MODELS_DIR / "best_model.joblib",
        MODELS_DIR / "best_model.pkl",
    ]
    model_path = next((p for p in model_candidates if p.exists()), model_candidates[0])
    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model missing. Run model training first."
        )
    model = joblib.load(model_path)

    summary_path = MODELS_DIR / "best_model.json"
    model_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    return metadata, model, model_summary


def load_median_artifacts():
    """Load legacy median artefacts (preserved for compatibility with older tooling)."""
    # Since we removed the suburb median module, we just return
    # a compatible interface with None values
    # The median is now calculated directly in preprocessing
    return None, None, {}

def load_cleaned_data() -> pd.DataFrame:
    candidates = [
        PREPROCESS_DIR / "derived.parquet",
        PREPROCESS_DIR / "derived.csv",
    ]
    data_path = next((p for p in candidates if p.exists()), candidates[0])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data missing ({PREPROCESS_DIR}). Run preprocessing before viewing analytics."
        )
    return load_parquet_or_csv(data_path)

def load_preprocess_metadata() -> Dict[str, Any]:
    metadata_path = PREPROCESS_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Preprocessing metadata not available yet.")
    return json.loads(metadata_path.read_text())


def load_training_sets() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    X_candidates = [
        TRAINING_DIR / "X.parquet",
        TRAINING_DIR / "X.csv",
    ]
    y_candidates = [
        TRAINING_DIR / "y.parquet",
        TRAINING_DIR / "y.csv",
    ]
    X_path = next((p for p in X_candidates if p.exists()), X_candidates[0])
    y_path = next((p for p in y_candidates if p.exists()), y_candidates[0])
    metadata_path = TRAINING_DIR / "feature_metadata.json"

    if not X_path.exists() or not y_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Training features not found. Run feature selection to generate X/y data files."
        )

    X = load_parquet_or_csv(X_path)
    y_df = load_parquet_or_csv(y_path)
    metadata = json.loads(metadata_path.read_text())
    target_col = metadata["target"]
    return X, y_df[target_col], metadata


def load_feature_importances() -> pd.DataFrame:
    path = TRAINING_DIR / "feature_importances.json"
    if not path.exists():
        raise FileNotFoundError(
            "Feature importance metadata missing. Run feature selection to regenerate it."
        )
    data = json.loads(path.read_text())
    return pd.DataFrame(data)


def load_model_metrics() -> pd.DataFrame:
    path = MODELS_DIR / "model_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            "Model metrics missing. Run model training to evaluate candidate estimators."
        )
    data = json.loads(path.read_text())
    return pd.DataFrame(data)


def load_street_coordinates() -> pd.DataFrame:
    path = STREET_COORDS_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Street coordinate mapping missing. Add config/street_coordinates.csv with suburb, street, latitude, longitude."
        )
    df = pd.read_csv(path)
    df.columns = [str(col).lower().strip() for col in df.columns]
    required = {"suburb", "street", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Street coordinate mapping must contain columns: suburb, street, latitude, longitude."
        )
    df["suburb"] = df["suburb"].astype(str).str.strip().str.upper()
    df["street"] = df["street"].astype(str).str.strip().str.title()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    return df


def read_yaml_config(path: Path) -> Dict[str, Any]:
    return load_config(path)


def write_yaml_config(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def load_yaml_bundle() -> Dict[str, Dict[str, Any]]:
    return {
        "preprocessing": read_yaml_config(PREPROCESS_CONFIG_PATH),
        "feature_engineering": read_yaml_config(FEATURE_ENGINEERING_CONFIG_PATH),
        "model": read_yaml_config(MODEL_CONFIG_PATH),
    }


def list_raw_files(path: Optional[Path] = None, pattern: str = "*.csv") -> List[Path]:
    base = path or DATA_DIR
    return sorted(base.glob(pattern))


def preview_raw_files(
    base_path: Path,
    pattern: str,
    include_columns: Optional[List[str]] = None,
    max_rows_per_file: int = 500,
) -> pd.DataFrame:
    csv_paths = sorted(base_path.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(
            f"No raw CSV files found in {base_path} matching pattern '{pattern}'."
        )

    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
                usecols=include_columns if include_columns else None,
                nrows=max_rows_per_file,
            )
        except ValueError:
            df = pd.read_csv(path, encoding="utf-8-sig", nrows=max_rows_per_file)
            if include_columns:
                existing = [col for col in include_columns if col in df.columns]
                if existing:
                    df = df[existing]
        df["__source_file"] = path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
