import yaml

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    load_cleaned_data,
    load_feature_importances,
    load_training_sets,
    read_yaml_config,
    write_yaml_config,
)

from src.config import FEATURE_ENGINEERING_CONFIG_PATH
from src.feature_selection import run_feature_selection


st.set_page_config(page_title="Feature Engineering", layout="wide")
st.title("Feature Engineering & Selection")
st.caption(
    "Tune the target variable, prune low-value features, and regenerate training matrices."
)


def _load_config_text() -> str:
    config = read_yaml_config(FEATURE_ENGINEERING_CONFIG_PATH)
    return yaml.safe_dump(config or {}, sort_keys=False)


if "feature_config_editor" not in st.session_state:
    st.session_state["feature_config_editor"] = _load_config_text()

config_text = st.text_area(
    "Feature engineering configuration (YAML)",
    height=340,
    key="feature_config_editor",
)


def _parse_config(text: str):
    try:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("Configuration root must be a mapping")
        return data, None
    except yaml.YAMLError as exc:
        return None, exc


config_data, parse_error = _parse_config(config_text)
if parse_error:
    st.error(f"YAML parsing error: {parse_error}")

st.markdown("### Guided adjustments")
try:
    cleaned_df = load_cleaned_data()
except FileNotFoundError:
    cleaned_df = pd.DataFrame()

with st.form("feature_form"):
    available_numeric = (
        cleaned_df.select_dtypes(include=[np.number]).columns.tolist() if not cleaned_df.empty else []
    )
    current_target = (config_data or {}).get("target", "priceFactor") if not parse_error else "priceFactor"
    target_choice = st.selectbox(
        "Target variable",
        options=[current_target] + [col for col in available_numeric if col != current_target],
        index=0,
    )

    current_threshold = float((config_data or {}).get("correlation_threshold", 0.9)) if not parse_error else 0.9
    threshold_choice = st.slider(
        "Correlation threshold",
        min_value=0.3,
        max_value=0.99,
        value=float(round(current_threshold, 2)),
        step=0.01,
    )

    drop_defaults = (
        [col for col in (config_data or {}).get("drop_columns", []) if isinstance(col, str)]
        if not parse_error
        else []
    )
    drop_selection = st.multiselect(
        "Columns to drop",
        options=sorted(cleaned_df.columns.tolist()) if not cleaned_df.empty else drop_defaults,
        default=drop_defaults,
    )

    force_defaults = (
        [col for col in (config_data or {}).get("force_keep", []) if isinstance(col, str)]
        if not parse_error
        else []
    )
    force_selection = st.multiselect(
        "Columns to force-keep",
        options=sorted(cleaned_df.columns.tolist()) if not cleaned_df.empty else force_defaults,
        default=force_defaults,
    )

    submitted = st.form_submit_button("Apply to YAML")

    if submitted:
        updated = dict(config_data or {})
        updated["target"] = target_choice
        updated["correlation_threshold"] = float(round(threshold_choice, 4))
        updated["drop_columns"] = sorted(set(drop_selection))
        updated["force_keep"] = sorted(set(force_selection))
        st.session_state["feature_config_editor"] = yaml.safe_dump(
            updated, sort_keys=False
        )
        st.experimental_rerun()


save_col, run_col = st.columns(2)
with save_col:
    if st.button("💾 Save configuration") and not parse_error:
        write_yaml_config(FEATURE_ENGINEERING_CONFIG_PATH, config_data)
        st.success("Configuration saved.")

with run_col:
    if st.button("🏗️ Run feature engineering"):
        if parse_error:
            st.error(f"Cannot run due to YAML error: {parse_error}")
        else:
            with st.spinner("Running feature selection..."):
                metadata = run_feature_selection(config_data)
            st.success(
                f"Feature selection completed with {len(metadata['selected_features'])} features."
            )
            st.session_state["feature_config_editor"] = yaml.safe_dump(
                config_data, sort_keys=False
            )


st.markdown("### Selected features")
try:
    X, y, metadata = load_training_sets()
except FileNotFoundError:
    st.info("Run feature engineering to populate training matrices.")
else:
    selected_features = metadata.get("selected_features", [])
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])

    cols = st.columns(3)
    cols[0].metric("Total features", len(selected_features))
    cols[1].metric("Numeric", len(numeric_features))
    cols[2].metric("Categorical", len(categorical_features))

    st.markdown("#### Feature lists")
    st.write(", ".join(selected_features) if selected_features else "No features selected")

    st.markdown("#### Target summary")
    st.write(f"Target: **{metadata.get('target')}**")
    st.write(f"Target type: **{metadata.get('target_type', 'price_factor')}**")

    st.markdown("#### Correlation heatmap")
    numeric_cols = [col for col in numeric_features if col in X.columns]
    if numeric_cols:
        corr_matrix = X[numeric_cols].corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap="Oranges"))
    else:
        st.info("No numeric features available for correlation heatmap.")


st.markdown("### Feature importance")
try:
    importance_df = load_feature_importances()
except FileNotFoundError:
    st.info("Feature importance becomes available after feature engineering.")
else:
    if importance_df.empty:
        st.info("Feature importance table is empty. Rerun feature selection.")
    else:
        importance_df = importance_df.sort_values("importance", ascending=False)
        st.bar_chart(importance_df.set_index("feature").head(20))
        st.dataframe(importance_df)
