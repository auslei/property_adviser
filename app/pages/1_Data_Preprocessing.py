from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import json

from property_adviser.common.app_utils import (
    load_cleaned_data,
    load_training_sets,
    preview_raw_files,
    read_yaml_config,
)

from property_adviser.common.app_utils import PREPROCESS_CONFIG_PATH, PREPROCESS_DIR
from property_adviser.preprocess import preprocess

from property_adviser.common.config import load_config

pp_cfg = load_config(Path(PREPROCESS_CONFIG_PATH))
DERIVED_OUTPUT_PATH = pp_cfg['derived_path']


st.set_page_config(page_title="Data Preprocessing", layout="wide")
st.title("Data Preprocessing")
st.caption(
    "Run the full preprocessing pipeline to generate all datasets, then inspect raw data, cleaned data, and derived data."
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
def _load_derived() -> pd.DataFrame:
    if DERIVED_OUTPUT_PATH.exists():
        return pd.read_parquet(DERIVED_OUTPUT_PATH)
    else:
        raise FileNotFoundError(f"Derived dataset not found at {DERIVED_OUTPUT_PATH}. Please run the preprocessing pipeline.")


@st.cache_data
def _load_metadata() -> dict:
    metadata_path = PREPROCESS_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please run the preprocessing pipeline.")


def run_full_preprocessing(config: Dict[str, object]) -> tuple[Path, Path, int]:
    # Run the full preprocessing pipeline
    output_path = preprocess(config)
    
    # Clear all caches
    _load_cleaned.clear()
    _load_raw_preview.clear()
    _load_derived.clear()
    _load_metadata.clear()
    
    # Get row count for derived data
    try:
        derived_df = pd.read_parquet(DERIVED_OUTPUT_PATH)
        row_count = derived_df.shape[0]
    except FileNotFoundError:
        row_count = 0
    
    return output_path, DERIVED_OUTPUT_PATH, row_count


def display_profile(df: pd.DataFrame, label: str) -> None:
    st.subheader(f"Profile: {label}")
    
    # Separate numerical and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numerical columns statistics
    if numeric_cols:
        st.markdown("### Numerical Columns")
        numeric_stats = []
        for col in numeric_cols:
            series = df[col].dropna()
            if not series.empty:
                stats = {
                    'Column': col,
                    'Mean': series.mean(),
                    'Median': series.median(),
                    'Max': series.max(),
                    'Std Dev': series.std()
                }
                numeric_stats.append(stats)
        
        if numeric_stats:
            numeric_df = pd.DataFrame(numeric_stats)
            # Format numbers for better readability
            for col in ['Mean', 'Median', 'Max', 'Std Dev']:
                if col in numeric_df.columns:
                    numeric_df[col] = numeric_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
            st.dataframe(numeric_df, hide_index=True)
        else:
            st.info("No numerical columns with data found.")
    else:
        st.info("No numerical columns found.")
    
    # Categorical columns frequencies
    if categorical_cols:
        st.markdown("### Categorical Columns")
        for col in categorical_cols:
            series = df[col].dropna()
            if not series.empty:
                with st.expander(f"{col} ({len(series.value_counts())} unique values)"):
                    value_counts = series.value_counts().reset_index()
                    value_counts.columns = ['Value', 'Count']
                    value_counts['Percentage'] = (value_counts['Count'] / len(series) * 100).round(2)
                    value_counts['Percentage'] = value_counts['Percentage'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(value_counts, hide_index=True)
    else:
        st.info("No categorical columns found.")


def get_available_years(df: pd.DataFrame) -> list:
    """Get available years from the dataset"""
    if 'saleYear' in df.columns:
        years = sorted(df['saleYear'].dropna().unique().tolist())
        return years
    return []


config_data = read_yaml_config(PREPROCESS_CONFIG_PATH)
data_cfg = config_data.get("data_source", {}) if isinstance(config_data, dict) else {}
base_path = Path(data_cfg.get("path", "data"))
pattern = data_cfg.get("pattern", "*.csv")
include_columns = data_cfg.get("include_columns")

# Run preprocessing button
if st.button("🔄 Run Full Preprocessing Pipeline"):
    with st.spinner("Running full preprocessing pipeline..."):
        try:
            cleaned_path, derived_path, row_count = run_full_preprocessing(config_data)
            st.success(f"Preprocessing completed successfully!")
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")


# Always show these three options
dataset_choice = st.selectbox(
    "Dataset stage",
    (
        "Raw data",
        "Cleaned data", 
        "Derived data"
    )
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


def display_metadata_summary(metadata: dict) -> None:
    st.subheader("Preprocessing Metadata")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{metadata.get('rows', 'N/A'):,}")
    with col2:
        st.metric("Total Columns", len(metadata.get('columns', [])))
    with col3:
        sources = metadata.get('source_files', {})
        st.metric("Source Files", len(sources))
    
    # Configuration summary
    with st.expander("Configuration Details"):
        config = metadata.get('configuration', {})
        
        if config:
            # Data source info
            data_source = config.get('data_source', {})
            if data_source:
                st.markdown("**Data Source Configuration:**")
                st.markdown(f"- Path: `{data_source.get('path', 'N/A')}`")
                st.markdown(f"- Pattern: `{data_source.get('pattern', 'N/A')}`")
                include_cols = data_source.get('include_columns', [])
                if include_cols:
                    st.markdown(f"- Include columns: {', '.join(include_cols)}")
            
            # Cleaning info
            cleaning = config.get('cleaning', {})
            if cleaning:
                st.markdown("**Cleaning Configuration:**")
                required_cols = cleaning.get('required_columns', [])
                if required_cols:
                    st.markdown(f"- Required columns: {', '.join(required_cols)}")
                st.markdown(f"- Primary target: `{cleaning.get('primary_target', 'N/A')}`")
                st.markdown(f"- Min comparable count: `{cleaning.get('min_comparable_count', 'N/A')}`")
            
            # Derivations info
            derivations = config.get('derivations', {})
            if derivations:
                st.markdown("**Derivations Configuration:**")
                for key, value in derivations.items():
                    if isinstance(value, dict):
                        enabled = value.get('enabled', True)
                        st.markdown(f"- {key}: {'✅ Enabled' if enabled else '❌ Disabled'}")


try:
    metadata = None
    try:
        metadata = _load_metadata()
    except FileNotFoundError:
        pass
    
    # Load the selected dataset
    selected_df = None
    if dataset_choice == "Raw data":
        selected_df = _load_raw_preview(base_path, pattern, include_columns)
    elif dataset_choice == "Cleaned data":
        selected_df = _load_cleaned()
    elif dataset_choice == "Derived data":
        selected_df = _load_derived()
    
    # Year filter
    available_years = get_available_years(selected_df) if selected_df is not None else []
    # Reverse sort years and put "All years" at the top
    year_options = ["All years"] + sorted(available_years, reverse=True) if available_years else ["All years"]
    selected_year = st.selectbox("Filter by year", year_options, index=0)
    
    # Apply year filter if a specific year is selected
    filtered_df = selected_df
    if selected_df is not None and selected_year != "All years" and 'saleYear' in selected_df.columns:
        filtered_df = selected_df[selected_df['saleYear'] == selected_year]
    
    # Show tabs
    if selected_df is not None:
        tab1, tab2 = st.tabs(["General", "Profile"])
        
        with tab1:
            # Show sample data
            if dataset_choice == "Raw data":
                display_frame(filtered_df, "Raw preview")
            elif dataset_choice == "Cleaned data":
                display_frame(filtered_df, "Cleaned dataset")
            elif dataset_choice == "Derived data":
                display_frame(filtered_df, "Derived dataset")
            
            # Show preprocessing status
            st.divider()
            if metadata:
                display_metadata_summary(metadata)
            else:
                st.info("Run the preprocessing pipeline to generate metadata.")
            
        with tab2:
            if filtered_df is not None and not filtered_df.empty:
                display_profile(filtered_df, f"{dataset_choice} ({selected_year})" if selected_year != "All years" else dataset_choice)
            elif filtered_df is not None and filtered_df.empty:
                st.info("No data available for the selected year.")
            else:
                st.info("No data available.")
    elif metadata:
        # Show metadata even if no data is loaded
        display_metadata_summary(metadata)
    else:
        st.info("Run the preprocessing pipeline to generate datasets and metadata.")
        
except FileNotFoundError as exc:
    st.warning(str(exc))
