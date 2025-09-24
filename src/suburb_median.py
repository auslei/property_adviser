import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import PREPROCESS_DIR, TRAINING_DIR


HISTORY_FILENAME = "suburb_month_medians.parquet"
GLOBAL_SUBURB_KEY = "__GLOBAL__"


def compute_baseline_medians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute baseline medians by suburb/year/month for feature engineering.
    
    This simplified approach computes observed historical medians that can be
    used as denominators for price factor calculations (salePrice / baselineMedian).
    """
    required = {"suburb", "salePrice", "saleYear", "saleMonth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for median computation: {sorted(missing)}")
    
    working = df.copy()
    working = working.dropna(subset=["salePrice", "saleYear", "saleMonth"])
    working["suburb"] = working["suburb"].fillna("Unknown")
    
    if working.empty:
        raise ValueError("No rows available to compute suburb/month medians.")
    
    # Compute medians by suburb/year/month
    suburb_medians = (
        working.groupby(["suburb", "saleYear", "saleMonth"])["salePrice"]
        .agg(medianPrice="median", transactionCount="size")
        .reset_index()
    )
    
    # Compute global medians for fallback when suburb data is unavailable
    global_medians = (
        working.groupby(["saleYear", "saleMonth"])["salePrice"]
        .agg(medianPrice="median", transactionCount="size")
        .reset_index()
    )
    global_medians["suburb"] = GLOBAL_SUBURB_KEY
    
    # Combine suburb and global medians
    combined_medians = pd.concat([suburb_medians, global_medians], ignore_index=True, sort=False)
    
    # Ensure proper data types
    combined_medians["saleYear"] = combined_medians["saleYear"].astype(int)
    combined_medians["saleMonth"] = combined_medians["saleMonth"].astype(int)
    combined_medians["transactionCount"] = combined_medians["transactionCount"].astype(int)
    combined_medians["medianPrice"] = combined_medians["medianPrice"].astype(float)
    
    # Sort for consistent ordering
    combined_medians = combined_medians.sort_values(["suburb", "saleYear", "saleMonth"]).reset_index(drop=True)
    
    return combined_medians


def prepare_suburb_median_artifacts(force: bool = False) -> Dict[str, object]:
    """
    Prepare and save baseline medians for feature engineering.
    
    This replaces the complex forecasting model with simple historical median computation.
    This function maintains the same interface as the original for backward compatibility.
    """
    from .preprocess import PREPROCESS_DIR
    
    # Load cleaned data
    data_path = PREPROCESS_DIR / "cleaned.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            "Preprocessed data not found. Run preprocessing first."
        )
    df = pd.read_parquet(data_path)
    
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    history_path = TRAINING_DIR / HISTORY_FILENAME
    
    # Skip if already exists and not forcing regeneration
    if not force and history_path.exists():
        history = pd.read_parquet(history_path)
        return {
            "history_rows": int(history.shape[0]),
            "suburbs": sorted(history[history["suburb"] != GLOBAL_SUBURB_KEY]["suburb"].unique()),
            "observed_months": sorted(history["saleYear"].astype(str) + "-" + history["saleMonth"].astype(str)),
        }
    
    # Compute baseline medians
    history = compute_baseline_medians(df)
    
    # Save to parquet file
    history.to_parquet(history_path, index=False)
    
    return {
        "history_rows": int(history.shape[0]),
        "suburbs": sorted(history[history["suburb"] != GLOBAL_SUBURB_KEY]["suburb"].unique()),
        "observed_months": sorted(history["saleYear"].astype(str) + "-" + history["saleMonth"].astype(str)),
    }


def load_baseline_median_history() -> pd.DataFrame:
    """
    Load baseline median history for feature engineering.
    
    This replaces the complex loading of model artifacts with simple parquet loading.
    """
    history_path = TRAINING_DIR / HISTORY_FILENAME
    if not history_path.exists():
        raise FileNotFoundError(
            "Baseline median history missing. Run preprocessing to generate baseline medians."
        )
    return pd.read_parquet(history_path)
