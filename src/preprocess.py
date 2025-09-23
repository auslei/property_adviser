import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    MIN_NON_NULL_FRACTION,
    PREPROCESS_CONFIG_PATH,
    PREPROCESS_DIR,
    RAW_DATA_PATTERN,
)
from .data_tracking import update_metadata_with_sources
from .configuration import load_yaml


SPECIAL_TO_ASCII = {
    "²": "2",
}

UNIT_KEYWORDS = (
    "unit",
    "flat",
    "apartment",
    "villa",
    "studio",
    "duplex",
)

HOUSE_KEYWORDS = (
    "house",
    "detached",
    "dwelling",
    "residence",
    "townhouse",
)


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


def simplify_property_type(raw_value: object) -> str:
    if not isinstance(raw_value, str):
        return ""
    value = _normalize_text(raw_value).strip().lower()
    for keyword in UNIT_KEYWORDS:
        if keyword in value:
            return "Unit"
    for keyword in HOUSE_KEYWORDS:
        if keyword in value:
            return "House"
    return ""


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
        .replace({
            "": np.nan,
            "-": np.nan,
            "nan": np.nan,
            "NaT": np.nan,
            "None": np.nan,
        })
    )


def bucket_categorical(series: pd.Series, min_count: int) -> pd.Series:
    value_counts = series.value_counts(dropna=False)
    allowed = set(value_counts[value_counts >= min_count].index)
    return series.apply(
        lambda value: value if value in allowed or value == "Unknown" else "Other"
    )


def _load_preprocess_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is not None:
        return config
    return load_yaml(PREPROCESS_CONFIG_PATH)


def _resolve_data_paths(config: Dict[str, Any]) -> List[Path]:
    data_cfg = config.get("data_source", {})
    base_path = Path(data_cfg.get("path", DATA_DIR))
    pattern = data_cfg.get("pattern", RAW_DATA_PATTERN)
    return sorted(base_path.glob(pattern))


def _filter_columns(df: pd.DataFrame, include: Optional[List[str]]) -> pd.DataFrame:
    if not include:
        return df
    existing = [col for col in include if col in df.columns]
    if not existing:
        return df
    return df.loc[:, existing]


def _apply_category_mappings(df: pd.DataFrame, mappings: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
    if not mappings:
        return df

    def _normalise(value: object) -> str:
        if not isinstance(value, str):
            return ""
        return _normalize_text(value).strip().lower()

    for column, groups in mappings.items():
        if column not in df.columns:
            continue
        lookup: Dict[str, str] = {}
        for label, keywords in groups.items():
            if isinstance(keywords, (list, tuple)):
                tokens = keywords
            else:
                tokens = [keywords]
            for token in tokens:
                key = _normalise(token)
                if key:
                    lookup[key] = label

        def _replace(value: object) -> object:
            if not isinstance(value, str):
                return value
            key = _normalise(value)
            return lookup.get(key, value)

        df[column] = df[column].apply(_replace)
    return df


def _derive_postcode_prefix(
    df: pd.DataFrame,
    source: str,
    length: int,
    output: str,
) -> pd.DataFrame:
    if source not in df.columns or output in df.columns:
        return df

    def _prefix(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return text[:length]

    df[output] = df[source].apply(_prefix)
    return df


def _apply_group_aggregate(
    df: pd.DataFrame,
    spec: Dict[str, Any],
) -> pd.DataFrame:
    if not spec.get("enabled", True):
        return df
    group_by = spec.get("group_by") or []
    target = spec.get("target")
    if not group_by or target is None:
        return df
    if any(col not in df.columns for col in group_by + [target]):
        return df
    agg_method = spec.get("aggregate", "median")
    output = spec.get("output") or f"{'_'.join(group_by)}_{agg_method}_{target}"
    if output in df.columns:
        return df
    aggregated = (
        df[group_by + [target]]
        .groupby(group_by, dropna=False)[target]
        .agg(agg_method)
        .reset_index(name=output)
    )
    return df.merge(aggregated, on=group_by, how="left")


def preprocess(config: Optional[Dict[str, Any]] = None) -> Path:
    resolved_config = _load_preprocess_config(config)

    data_cfg = resolved_config.get("data_source", {})
    base_path = Path(data_cfg.get("path", DATA_DIR))
    pattern = data_cfg.get("pattern", RAW_DATA_PATTERN)
    include_columns = data_cfg.get("include_columns")
    encoding = data_cfg.get("encoding", "utf-8-sig")

    csv_paths = _resolve_data_paths(resolved_config)
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {base_path} matching pattern '{pattern}'."
        )

    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                usecols=include_columns if include_columns else None,
            )
        except ValueError:
            df = pd.read_csv(path, encoding=encoding)
            df = _filter_columns(df, include_columns)
        else:
            df = _filter_columns(df, include_columns)
        df["__source_file"] = path.name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    cleaned_names: Dict[str, int] = {}
    column_mapping: Dict[str, str] = {}
    renamed_columns: List[str] = []
    for col in combined.columns:
        new_name = clean_column_name(col, cleaned_names)
        renamed_columns.append(new_name)
        column_mapping[col] = new_name
    combined.columns = renamed_columns

    if include_columns:
        allowed_names = {column_mapping.get(col, col) for col in include_columns}
        allowed_names |= {"__source_file"}
        keep = [col for col in combined.columns if col in allowed_names]
        if keep:
            combined = combined.loc[:, keep]

    combined = remove_mostly_empty_columns(combined)

    derivations_cfg = resolved_config.get("derivations", {})
    street_spec = derivations_cfg.get("street", {})
    street_source = street_spec.get("source", "streetAddress")
    street_output = street_spec.get("output", "street")
    if street_source in combined.columns:
        combined[street_output] = combined[street_source].apply(extract_street)
        if street_spec.get("drop_unknown", False):
            combined = combined[combined[street_output] != "Unknown"].copy()

    if "propertyType" in combined.columns:
        simplified = combined["propertyType"].apply(simplify_property_type)
        combined["propertyType"] = simplified
        combined = combined[combined["propertyType"] != ""].copy()

    cleaning_cfg = resolved_config.get("cleaning", {})
    category_mappings = cleaning_cfg.get("category_mappings", {})
    combined = _apply_category_mappings(combined, category_mappings)

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

    required_columns = cleaning_cfg.get("required_columns", [])
    for col in required_columns:
        if col in combined.columns:
            combined = combined[combined[col].notna()].copy()

    target_col = cleaning_cfg.get("primary_target", "salePrice")
    if target_col in combined.columns:
        combined = combined[combined[target_col].notna()].copy()

    comparable_cols = ["street", "propertyType", "bed", "bath", "car"]
    if all(col in combined.columns for col in comparable_cols):
        key_frame = combined[comparable_cols].copy()

        def _value_token(value: object) -> str:
            if pd.isna(value):
                return "Unknown"
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            if isinstance(value, float):
                if np.isfinite(value) and float(value).is_integer():
                    return str(int(value))
                return f"{value:.2f}"
            return str(value).strip() or "Unknown"

        keys = key_frame.apply(lambda row: "|".join(_value_token(val) for val in row), axis=1)
        counts = keys.value_counts()
        combined["comparableCount"] = keys.map(counts).fillna(0).astype(int)
        combined = combined[combined["comparableCount"] >= 2].copy()

    street_mean_spec = derivations_cfg.get("street_year_median_price")
    if street_mean_spec:
        combined = _apply_group_aggregate(combined, street_mean_spec)

    postcode_spec = derivations_cfg.get("postcode_prefix")
    if postcode_spec and postcode_spec.get("enabled", True):
        source_col = postcode_spec.get("source", "postcode")
        output_col = postcode_spec.get("output", "postcodePrefix")
        length = int(postcode_spec.get("length", 1))
        combined = _derive_postcode_prefix(combined, source_col, length, output_col)

    numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if target_col and col == target_col:
            continue
        median = combined[col].median()
        combined[col] = combined[col].fillna(median)

    categorical_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        combined[col] = combined[col].fillna("Unknown")

    bucket_rules = {
        "agency": 20,
        "landUse": 15,
        "propertyType": 15,
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
        "configuration": resolved_config,
    }
    summary = update_metadata_with_sources(summary)
    metadata_path = PREPROCESS_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2))

    return output_path


if __name__ == "__main__":
    path = preprocess()
    print(f"Preprocessed data saved to {path}")
