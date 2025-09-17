import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    MIN_NON_NULL_FRACTION,
    PREPROCESS_DIR,
    RAW_DATA_PATTERN,
)
from .data_tracking import update_metadata_with_sources


SPECIAL_TO_ASCII = {
    "²": "2",
}


def _normalize_text(value: str) -> str:
    normalized = value
    for src, dst in SPECIAL_TO_ASCII.items():
        normalized = normalized.replace(src, dst)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized


def clean_column_name(name: str, existing: Dict[str, int]) -> str:
    name = _normalize_text(str(name)).strip()
    if not name:
        name = "column"
    tokens = re.split(r"[^0-9a-zA-Z]+", name)
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        tokens = ["column"]
    base = tokens[0].lower() + "".join(tok.capitalize() for tok in tokens[1:])
    count = existing.get(base, 0)
    if count:
        new_name = f"{base}{count+1}"
    else:
        new_name = base
    existing[base] = count + 1
    return new_name


def extract_street(address: str) -> str:
    if not isinstance(address, str):
        return "Unknown"
    cleaned = address.strip()
    if not cleaned or cleaned == "-":
        return "Unknown"
    cleaned = _normalize_text(cleaned.upper())
    cleaned = cleaned.replace("  ", " ")
    cleaned = re.split(r"\s+", cleaned)
    if not cleaned:
        return "Unknown"
    joined = " ".join(cleaned)
    # remove unit numbers like 3/55 or 12A
    joined = re.sub(r"^[0-9]+[A-Z]?/[0-9]+\s+", "", joined)
    joined = re.sub(r"^[0-9]+[A-Z]?\s+", "", joined)
    joined = joined.strip()
    if not joined:
        return "Unknown"
    return joined.title()


def remove_mostly_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    non_null_fraction = df.notna().mean()
    keep_cols = non_null_fraction[non_null_fraction >= MIN_NON_NULL_FRACTION].index.tolist()
    return df.loc[:, keep_cols]


def coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[,$]", "", regex=True)
        .str.replace(r"[^0-9.-]+", "", regex=True)
        .replace("", np.nan)
        .replace("-", np.nan)
        .replace("nan", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_strings(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "-": np.nan, "nan": np.nan})
    )


def bucket_categorical(series: pd.Series, min_count: int) -> pd.Series:
    value_counts = series.value_counts(dropna=False)
    allowed = set(value_counts[value_counts >= min_count].index)
    return series.apply(
        lambda value: value if value in allowed or value == "Unknown" else "Other"
    )


def preprocess() -> Path:
    csv_paths = sorted(DATA_DIR.glob(RAW_DATA_PATTERN))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}.")

    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path, encoding="utf-8-sig")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    cleaned_names: Dict[str, int] = {}
    combined.columns = [clean_column_name(col, cleaned_names) for col in combined.columns]

    combined = remove_mostly_empty_columns(combined)

    if "streetAddress" in combined.columns:
        combined["street"] = combined["streetAddress"].apply(extract_street)

    numeric_candidates = [
        "bed",
        "bath",
        "car",
        "landSizeM2",
        "floorSizeM2",
        "yearBuilt",
        "salePrice",
        "postcode",
    ]
    for col in numeric_candidates:
        if col in combined.columns:
            combined[col] = coerce_numeric(combined[col])

    if "saleDate" in combined.columns:
        sale_date = pd.to_datetime(combined["saleDate"], errors="coerce")
        combined["saleDate"] = sale_date.dt.strftime("%Y%m")
        combined["saleYear"] = sale_date.dt.year
        combined["saleMonth"] = sale_date.dt.month

    for col in combined.columns:
        if combined[col].dtype == object:
            combined[col] = normalize_strings(combined[col])

    numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        median = combined[col].median()
        combined[col] = combined[col].fillna(median)

    categorical_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        combined[col] = combined[col].fillna("Unknown")

    bucket_rules = {
        "agency": 20,
        "landUse": 15,
        "propertyType": 15,
        "street": 10,
    }
    for col, threshold in bucket_rules.items():
        if col in combined.columns:
            combined[col] = bucket_categorical(combined[col], threshold)

    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREPROCESS_DIR / "cleaned.parquet"
    combined.to_parquet(output_path, index=False)

    summary = {
        "rows": int(combined.shape[0]),
        "columns": combined.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
    summary = update_metadata_with_sources(summary)
    metadata_path = PREPROCESS_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2))

    return output_path


if __name__ == "__main__":
    path = preprocess()
    print(f"Preprocessed data saved to {path}")
