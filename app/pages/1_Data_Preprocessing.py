from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from utils import (
    load_cleaned_data,
    load_training_sets,
    preview_raw_files,
    read_yaml_config,
)

from src.config import PREPROCESS_CONFIG_PATH, PREPROCESS_DIR
from src.preprocess import preprocess


DERIVED_OUTPUT_PATH = PREPROCESS_DIR / "derived.parquet"


st.set_page_config(page_title="Data Preprocessing", layout="wide")
st.title("Data Preprocessing")
st.caption(
    "Run cleansing and derivation steps on demand, then inspect raw, cleaned, derived, or feature datasets."
)


@st.cache_data
def _load_raw_preview(base_path: Path, pattern: str, include_columns: Optional[list[str]]) -> pd.DataFrame:
    return preview_raw_files(
        base_path=base_path,
        pattern=pattern,
        include_columns=include_columns,
        max_rows_per_file=500,
    )


@st.cache_data
def _load_cleaned() -> pd.DataFrame:
    return load_cleaned_data()


@st.cache_data
def _load_features() -> pd.DataFrame:
    X, _, _ = load_training_sets()
    return X


@st.cache_data
def _load_derived() -> pd.DataFrame:
    if DERIVED_OUTPUT_PATH.exists():
        return pd.read_parquet(DERIVED_OUTPUT_PATH)
    cleaned = load_cleaned_data()
    config = read_yaml_config(PREPROCESS_CONFIG_PATH)
    derived = build_derived_view(cleaned, config)
    return derived


def build_derived_view(cleaned: pd.DataFrame, config: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    if cleaned.empty:
        return pd.DataFrame()

    derivations = (config or {}).get("derivations", {})
    derived_columns: list[str] = []
    for spec in derivations.values():
        if isinstance(spec, dict):
            output = spec.get("output")
            if isinstance(output, str):
                derived_columns.append(output)

    context_columns = [col for col in ["suburb", "street", "saleYear", "saleMonth"] if col in cleaned.columns]

    seen: Dict[str, str] = {}

    def _unique_name(name: str) -> str:
        if name not in seen:
            seen[name] = name
            return name
        counter = 2
        candidate = f"{name}_{counter}"
        while candidate in seen:
            counter += 1
            candidate = f"{name}_{counter}"
        seen[candidate] = candidate
        return candidate

    view = pd.DataFrame()
    
    all_columns_to_process = context_columns + [c for c in derived_columns if c in cleaned.columns]

    for col in all_columns_to_process:
        new_name = _unique_name(col)
        view[new_name] = cleaned[col]

    if view.empty:
        return pd.DataFrame()

    return view.drop_duplicates()


def run_derivation_step(config: Dict[str, object]) -> int:
    cleaned = load_cleaned_data()
    derived = build_derived_view(cleaned, config)
    DERIVED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    derived.to_parquet(DERIVED_OUTPUT_PATH, index=False)
    _load_derived.clear()
    return derived.shape[0]


def run_cleansing_step(config: Dict[str, object]) -> Path:
    output = preprocess(config)
    _load_cleaned.clear()
    _load_raw_preview.clear()
    _load_derived.clear()
    return output


config_data = read_yaml_config(PREPROCESS_CONFIG_PATH)
data_cfg = config_data.get("data_source", {}) if isinstance(config_data, dict) else {}
base_path = Path(data_cfg.get("path", "data"))
pattern = data_cfg.get("pattern", "*.csv")
include_columns = data_cfg.get("include_columns")


controls_col, derivation_col = st.columns(2)
with controls_col:
    if st.button("⚙️ Run cleansing"):
        with st.spinner("Running cleansing pipeline..."):
            output_path = run_cleansing_step(config_data)
        st.success(f"Cleansed dataset refreshed at {output_path}.")

with derivation_col:
    if st.button("🧮 Compute derivations"):
        try:
            with st.spinner("Generating derived view..."):
                row_count = run_derivation_step(config_data)
        except FileNotFoundError as exc:
            st.error(str(exc))
        else:
            st.success(f"Derived dataset saved to {DERIVED_OUTPUT_PATH} ({row_count:,} rows).")


dataset_choice = st.selectbox(
    "Dataset stage",
    (
        "Raw data",
        "Cleaned data",
        "Derived data",
        "Feature matrix",
    ),
)


def display_frame(frame: pd.DataFrame, label: str) -> None:
    st.write(f"{label}: {frame.shape[0]:,} rows × {frame.shape[1]} columns")
    if frame.empty:
        st.info("No rows available.")
        return
    preview_limit = min(500, frame.shape[0])
    st.dataframe(frame.head(preview_limit))
    if frame.shape[0] > preview_limit:
        st.caption(f"Showing first {preview_limit} rows.")


try:
    if dataset_choice == "Raw data":
        raw_preview = _load_raw_preview(base_path, pattern, include_columns)
        display_frame(raw_preview, "Raw preview")
    elif dataset_choice == "Cleaned data":
        cleaned_df = _load_cleaned()
        display_frame(cleaned_df, "Cleaned dataset")
    elif dataset_choice == "Derived data":
        derived_df = _load_derived()
        display_frame(derived_df, "Derived dataset")
    else:
        features_df = _load_features()
        display_frame(features_df, "Feature matrix (X)")
except FileNotFoundError as exc:
    st.warning(str(exc))
