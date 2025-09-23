import yaml

import pandas as pd
import streamlit as st

from utils import (
    load_model_metrics,
    load_model_resources,
    load_training_sets,
    read_yaml_config,
    write_yaml_config,
)

from src.config import MODEL_CONFIG_PATH
from src.model_training import train_models


st.set_page_config(page_title="Model Selection", layout="wide")
st.title("Model Selection & Training")
st.caption(
    "Control candidate regressors, tune hyper-parameters, and persist the best-performing model."
)


def _load_config_text() -> str:
    config = read_yaml_config(MODEL_CONFIG_PATH)
    return yaml.safe_dump(config or {}, sort_keys=False)


if "model_config_editor" not in st.session_state:
    st.session_state["model_config_editor"] = _load_config_text()

config_text = st.text_area(
    "Model configuration (YAML)",
    height=340,
    key="model_config_editor",
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
    X, y, metadata = load_training_sets()
except FileNotFoundError:
    X = pd.DataFrame()
    metadata = {}

available_features = metadata.get("selected_features", []) if metadata else []

with st.form("model_form"):
    manual_cfg = (config_data or {}).get("manual_feature_adjustments", {}) if not parse_error else {}
    include_defaults = manual_cfg.get("include", []) if isinstance(manual_cfg, dict) else []
    exclude_defaults = manual_cfg.get("exclude", []) if isinstance(manual_cfg, dict) else []

    include_selection = st.multiselect(
        "Force-include features",
        options=available_features,
        default=[feat for feat in include_defaults if feat in available_features],
    )
    exclude_selection = st.multiselect(
        "Exclude features",
        options=available_features,
        default=[feat for feat in exclude_defaults if feat in available_features],
    )

    models_cfg = (config_data or {}).get("models", {}) if not parse_error else {}
    model_names = sorted(models_cfg.keys()) if models_cfg else []
    enabled_models = []
    if model_names:
        st.markdown("#### Enable/disable models")
        cols = st.columns(min(3, len(model_names)))
        for idx, name in enumerate(model_names):
            column = cols[idx % len(cols)]
            enabled = models_cfg.get(name, {}).get("enabled", True)
            enabled_models.append((name, column.checkbox(name, value=enabled)))

    submitted = st.form_submit_button("Apply to YAML")

    if submitted and not parse_error:
        updated = dict(config_data or {})
        updated["manual_feature_adjustments"] = {
            "include": sorted(set(include_selection)),
            "exclude": sorted(set(exclude_selection)),
        }
        if model_names:
            updated_models = dict(models_cfg)
            for name, enabled_flag in enabled_models:
                updated_models.setdefault(name, {})["enabled"] = bool(enabled_flag)
            updated["models"] = updated_models
        st.session_state["model_config_editor"] = yaml.safe_dump(
            updated, sort_keys=False
        )
        st.experimental_rerun()


save_col, run_col = st.columns(2)
with save_col:
    if st.button("💾 Save configuration") and not parse_error:
        write_yaml_config(MODEL_CONFIG_PATH, config_data)
        st.success("Configuration saved.")

with run_col:
    if st.button("🚀 Train models"):
        if parse_error:
            st.error(f"Cannot run training due to YAML error: {parse_error}")
        else:
            with st.spinner("Training models and performing grid search..."):
                summary = train_models(config_data)
            st.success(f"Training completed. Best model: {summary['best_model']}")
            st.session_state["model_config_editor"] = yaml.safe_dump(
                config_data, sort_keys=False
            )


st.markdown("### Persisted model summary")
try:
    metadata, model, model_summary = load_model_resources()
except FileNotFoundError as exc:
    st.info(str(exc))
else:
    cols = st.columns(3)
    cols[0].metric("Best model", model_summary.get("best_model", "N/A"))
    cols[1].metric("Validation R²", f"{model_summary.get('r2', float('nan')):.3f}")
    cols[2].metric("Target type", metadata.get("target_type", "price_factor"))
    st.markdown("#### Feature metadata snapshot")
    st.json(
        {
            "selected_features": metadata.get("selected_features", []),
            "manual_adjustments": model_summary.get("manual_adjustments", {}),
            "split": model_summary.get("split", {}),
        }
    )


st.markdown("### Model metrics")
try:
    metrics_df = load_model_metrics()
except FileNotFoundError:
    st.info("Training metrics will appear after running model selection.")
else:
    if metrics_df.empty:
        st.info("Metric table is empty. Rerun training.")
    else:
        formatted = metrics_df.copy()
        for column in ["mae", "rmse"]:
            if column in formatted.columns:
                formatted[column] = formatted[column].round(0)
        if "r2" in formatted.columns:
            formatted["r2"] = formatted["r2"].round(3)
        st.dataframe(formatted)
