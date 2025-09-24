import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.config import (
        DATA_DIR,
        FEATURE_ENGINEERING_CONFIG_PATH,
        MODEL_CONFIG_PATH,
        MODELS_DIR,
        PREPROCESS_CONFIG_PATH,
        PREPROCESS_DIR,
        STREET_COORDS_PATH,
        TRAINING_DIR,
    )
except ImportError:
    # Fallback when an older src.config is imported (e.g. cached Streamlit session).
    DATA_DIR = PROJECT_ROOT / "data"
    PREPROCESS_DIR = PROJECT_ROOT / "data_preprocess"
    TRAINING_DIR = PROJECT_ROOT / "data_training"
    MODELS_DIR = PROJECT_ROOT / "models"
    CONFIG_DIR = PROJECT_ROOT / "config"
    PREPROCESS_CONFIG_PATH = CONFIG_DIR / "preprocessing.yml"
    FEATURE_ENGINEERING_CONFIG_PATH = CONFIG_DIR / "features.yml"
    MODEL_CONFIG_PATH = CONFIG_DIR / "model.yml"
    STREET_COORDS_PATH = CONFIG_DIR / "street_coordinates.csv"

from src.configuration import load_yaml
from src.suburb_median import load_baseline_median_history


@st.cache_resource
def load_model_resources() -> Tuple[Dict[str, Any], object, Dict[str, Any]]:
    metadata_path = TRAINING_DIR / "feature_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Feature metadata missing. Run preprocessing and feature selection first."
        )
    metadata = json.loads(metadata_path.read_text())

    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model missing. Run model training first."
        )
    model = joblib.load(model_path)

    summary_path = MODELS_DIR / "best_model.json"
    model_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    return metadata, model, model_summary


@st.cache_resource
def load_median_artifacts():
    """
    Load median artifacts for the Streamlit app.
    
    This simplified version returns only the history since the ML forecasting model
    has been eliminated. Returns a compatible tuple interface:
    (history, None, {}) where None represents the eliminated model.
    """
    history = load_baseline_median_history()
    # Return compatible interface: (history, model, metadata)
    # Since we've eliminated the model, we return None for model and empty dict for metadata
    return history, None, {}


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    data_path = PREPROCESS_DIR / "cleaned.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            "Preprocessed data missing. Run preprocessing before viewing analytics."
        )
    return pd.read_parquet(data_path)


@st.cache_data
def load_preprocess_metadata() -> Dict[str, Any]:
    metadata_path = PREPROCESS_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Preprocessing metadata not available yet.")
    return json.loads(metadata_path.read_text())


@st.cache_data
def load_training_sets() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    X_path = TRAINING_DIR / "X.parquet"
    y_path = TRAINING_DIR / "y.parquet"
    metadata_path = TRAINING_DIR / "feature_metadata.json"

    if not X_path.exists() or not y_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Training features not found. Run feature selection to generate X/y parquet files."
        )

    X = pd.read_parquet(X_path)
    y_df = pd.read_parquet(y_path)
    metadata = json.loads(metadata_path.read_text())
    target_col = metadata["target"]
    return X, y_df[target_col], metadata


@st.cache_data
def load_feature_importances() -> pd.DataFrame:
    path = TRAINING_DIR / "feature_importances.json"
    if not path.exists():
        raise FileNotFoundError(
            "Feature importance metadata missing. Run feature selection to regenerate it."
        )
    data = json.loads(path.read_text())
    return pd.DataFrame(data)


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    path = MODELS_DIR / "model_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            "Model metrics missing. Run model training to evaluate candidate estimators."
        )
    data = json.loads(path.read_text())
    return pd.DataFrame(data)


@st.cache_data
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
    return load_yaml(path)


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
