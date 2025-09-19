import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import MODELS_DIR, PREPROCESS_DIR, RANDOM_STATE, TRAINING_DIR


HISTORY_FILENAME = "suburb_month_medians.parquet"
MODEL_FILENAME = "suburb_median_model.pkl"
META_FILENAME = "suburb_median_model_meta.json"
GLOBAL_SUBURB_KEY = "__GLOBAL__"


def _load_clean_data() -> pd.DataFrame:
    data_path = PREPROCESS_DIR / "cleaned.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            "Preprocessed data not found. Run preprocessing before building medians."
        )
    return pd.read_parquet(data_path)


def _compute_history(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    required = {"suburb", "salePrice", "saleDate", "saleYear", "saleMonth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for median aggregation: {sorted(missing)}")

    working = df.copy()
    working = working.dropna(subset=["salePrice", "saleDate", "saleYear", "saleMonth"])
    working["suburb"] = working["suburb"].fillna("Unknown")
    working["saleDate"] = working["saleDate"].astype(str)

    if working.empty:
        raise ValueError("No rows available to compute suburb/month medians.")

    group_cols = ["suburb", "saleYear", "saleMonth", "saleDate"]
    aggregated = (
        working.groupby(group_cols, dropna=False)["salePrice"]
        .agg(medianPrice="median", transactionCount="size")
        .reset_index()
    )

    overall_group = (
        working.groupby(["saleYear", "saleMonth", "saleDate"], dropna=False)["salePrice"]
        .agg(medianPrice="median", transactionCount="size")
        .reset_index()
    )
    overall_group["suburb"] = GLOBAL_SUBURB_KEY
    aggregated = pd.concat([aggregated, overall_group], ignore_index=True, sort=False)

    aggregated["saleYear"] = aggregated["saleYear"].astype(int)
    aggregated["saleMonth"] = aggregated["saleMonth"].astype(int)
    aggregated["transactionCount"] = aggregated["transactionCount"].astype(int)

    month_start = pd.to_datetime(
        aggregated["saleDate"].astype(str) + "01", format="%Y%m%d", errors="coerce"
    )
    if month_start.isna().all():
        raise ValueError("Unable to parse saleDate values into month-period timestamps.")

    base_date = month_start.min()
    base_year = int(base_date.year)
    base_month = int(base_date.month)

    aggregated["timeIndex"] = (
        (aggregated["saleYear"] - base_year) * 12
        + (aggregated["saleMonth"] - base_month)
    ).astype(int)
    aggregated["monthStart"] = month_start.dt.strftime("%Y-%m-%d")

    lookups = {
        "base_year": base_year,
        "base_month": base_month,
    }

    aggregated = aggregated.sort_values(["suburb", "saleYear", "saleMonth"]).reset_index(drop=True)
    return aggregated, lookups


def _train_forecaster(history: pd.DataFrame, base_info: Dict[str, int]) -> Tuple[Pipeline, Dict[str, object]]:
    feature_columns = ["suburb", "timeIndex", "monthSin", "monthCos", "transactionCount"]

    working = history.copy()
    working["monthSin"] = np.sin(2 * np.pi * working["saleMonth"] / 12.0)
    working["monthCos"] = np.cos(2 * np.pi * working["saleMonth"] / 12.0)

    preprocessor = ColumnTransformer(
        [
            (
                "suburb",
                OneHotEncoder(handle_unknown="ignore"),
                ["suburb"],
            )
        ],
        remainder="passthrough",
    )

    model = GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
    )

    pipeline = Pipeline([
        ("features", preprocessor),
        ("model", model),
    ])

    pipeline.fit(working[feature_columns], working["medianPrice"])

    metadata = {
        "base_year": base_info["base_year"],
        "base_month": base_info["base_month"],
        "max_time_index": int(working["timeIndex"].max()),
        "max_year": int(working.loc[working["timeIndex"].idxmax(), "saleYear"]),
        "max_month": int(working.loc[working["timeIndex"].idxmax(), "saleMonth"]),
        "feature_columns": feature_columns,
        "global_suburb_key": GLOBAL_SUBURB_KEY,
        "observed_months": sorted(
            set(working.loc[working["suburb"] != GLOBAL_SUBURB_KEY, "saleDate"])
        ),
        "suburbs": sorted(set(working["suburb"].unique())),
        "mean_transaction_count": float(working["transactionCount"].mean()),
    }

    return pipeline, metadata


def prepare_suburb_median_artifacts(force: bool = False) -> Dict[str, object]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    history_path = TRAINING_DIR / HISTORY_FILENAME
    model_path = MODELS_DIR / MODEL_FILENAME
    meta_path = MODELS_DIR / META_FILENAME

    if not force and history_path.exists() and model_path.exists() and meta_path.exists():
        history = pd.read_parquet(history_path)
        metadata = json.loads(meta_path.read_text())
        return {
            "history_rows": int(history.shape[0]),
            "suburbs": metadata.get("suburbs", []),
            "observed_months": metadata.get("observed_months", []),
        }

    df = _load_clean_data()
    history, base_info = _compute_history(df)

    history.to_parquet(history_path, index=False)

    pipeline, metadata = _train_forecaster(history, base_info)
    joblib.dump(pipeline, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2))

    return {
        "history_rows": int(history.shape[0]),
        "suburbs": metadata.get("suburbs", []),
        "observed_months": metadata.get("observed_months", []),
    }


def load_suburb_median_history() -> pd.DataFrame:
    history_path = TRAINING_DIR / HISTORY_FILENAME
    if not history_path.exists():
        raise FileNotFoundError(
            "Suburb/month median history missing. Run suburb median preparation first."
        )
    return pd.read_parquet(history_path)


def load_suburb_median_artifacts() -> Tuple[pd.DataFrame, Pipeline, Dict[str, object]]:
    history = load_suburb_median_history()
    model_path = MODELS_DIR / MODEL_FILENAME
    meta_path = MODELS_DIR / META_FILENAME
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Suburb median model artefacts missing. Run suburb median preparation first."
        )
    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text())
    return history, model, metadata


def _compute_time_index(year: int, month: int, base_year: int, base_month: int) -> int:
    return (year - base_year) * 12 + (month - base_month)


def estimate_suburb_median(
    suburb: str,
    year: int,
    month: int,
    history: pd.DataFrame,
    model: Pipeline,
    metadata: Dict[str, object],
) -> Optional[Dict[str, object]]:
    if not (1 <= month <= 12):
        raise ValueError("Month must be within 1..12")

    suburb_value = suburb or "Unknown"
    lookup = history[
        (history["suburb"] == suburb_value)
        & (history["saleYear"] == year)
        & (history["saleMonth"] == month)
    ]
    if not lookup.empty:
        row = lookup.iloc[0]
        return {
            "median": float(row["medianPrice"]),
            "source": "observed",
            "transaction_count": int(row["transactionCount"]),
            "suburb": suburb_value,
        }

    global_lookup = history[
        (history["suburb"] == GLOBAL_SUBURB_KEY)
        & (history["saleYear"] == year)
        & (history["saleMonth"] == month)
    ]
    if not global_lookup.empty():
        row = global_lookup.iloc[0]
        return {
            "median": float(row["medianPrice"]),
            "source": "global_observed",
            "transaction_count": int(row["transactionCount"]),
            "suburb": GLOBAL_SUBURB_KEY,
        }

    base_year = int(metadata["base_year"])
    base_month = int(metadata["base_month"])
    time_index = _compute_time_index(year, month, base_year, base_month)

    feature_columns = metadata.get("feature_columns", [])

    month_sin = math.sin(2 * math.pi * month / 12.0)
    month_cos = math.cos(2 * math.pi * month / 12.0)

    input_row = {
        "suburb": suburb_value
        if suburb_value in metadata.get("suburbs", [])
        else metadata.get("global_suburb_key", GLOBAL_SUBURB_KEY),
        "timeIndex": time_index,
        "monthSin": month_sin,
        "monthCos": month_cos,
        "transactionCount": metadata.get("mean_transaction_count", 1.0),
    }

    X = pd.DataFrame([input_row])[feature_columns]
    prediction = float(model.predict(X)[0])

    return {
        "median": prediction,
        "source": "forecast",
        "transaction_count": None,
        "suburb": input_row["suburb"],
    }
