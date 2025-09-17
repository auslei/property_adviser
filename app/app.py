import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
import streamlit as st

# Ensure the project root (which contains the src package) is on the Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import MODELS_DIR, PREPROCESS_DIR, TRAINING_DIR


@st.cache_resource
def load_resources() -> Tuple[Dict[str, object], object, Dict[str, object]]:
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


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    data_path = PREPROCESS_DIR / "cleaned.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            "Preprocessed data missing. Run preprocessing before viewing analytics."
        )
    return pd.read_parquet(data_path)


def render_inputs(metadata: Dict[str, object]) -> Dict[str, object]:
    categorical_features = metadata.get("categorical_features", [])
    numeric_features = metadata.get("numeric_features", [])
    categorical_levels = metadata.get("categorical_levels", {})
    numeric_summary = metadata.get("numeric_summary", {})

    inputs: Dict[str, object] = {}

    st.header("Property Attributes")

    if categorical_features:
        st.subheader("Categorical")
    for feature in categorical_features:
        options = categorical_levels.get(feature, [])
        if not options:
            options = ["Unknown"]
        default_value = options[0]
        inputs[feature] = st.selectbox(
            feature,
            options,
            index=options.index(default_value) if default_value in options else 0,
        )

    if numeric_features:
        st.subheader("Numerical")
    for feature in numeric_features:
        stats = numeric_summary.get(feature, {})
        min_val = float(stats.get("min", 0.0))
        max_val = float(stats.get("max", max(min_val + 1.0, 1.0)))
        median = float(stats.get("median", (min_val + max_val) / 2 if max_val > min_val else min_val))
        if max_val <= min_val:
            max_val = min_val + max(abs(min_val), 1.0)
        step = 1.0
        if max_val - min_val < 10:
            step = 0.5
        if max_val - min_val < 2:
            step = 0.1
        inputs[feature] = st.number_input(
            feature,
            value=median,
            min_value=min_val,
            max_value=max_val,
            step=step,
        )

    return inputs


def main():
    st.set_page_config(page_title="Property Price Predictor", layout="centered")
    st.title("Property Price Predictor")
    st.markdown(
        "Select the property attributes to predict the expected sale price for the coming year."
    )

    metadata, model, model_summary = load_resources()

    inputs = render_inputs(metadata)

    if st.button("Predict Price"):
        input_df = pd.DataFrame([inputs])
        prediction = float(model.predict(input_df)[0])
        st.success(f"Estimated price next year: ${prediction:,.0f}")

    if model_summary:
        st.sidebar.header("Model Summary")
        st.sidebar.write(f"Best model: {model_summary.get('best_model', 'N/A')}")
        if "r2" in model_summary:
            st.sidebar.write(f"Validation R²: {model_summary['r2']:.3f}")
        st.sidebar.write(f"Model file: {model_summary.get('model_path', 'N/A')}")

    st.sidebar.caption("Ensure preprocessing and training scripts have been run before using the app.")

    st.divider()
    st.header("Kew Price Hotspots")

    try:
        cleaned_df = load_cleaned_data()
    except FileNotFoundError as exc:
        st.info(str(exc))
    else:
        kew_df = cleaned_df[cleaned_df["suburb"].str.upper() == "KEW"]
        if kew_df.empty:
            st.info("No property records for Kew are available in the current dataset.")
        else:
            street_prices = (
                kew_df.assign(street=kew_df["street"].fillna("Unknown"))
                .groupby("street", dropna=False)["salePrice"]
                .mean()
                .sort_values(ascending=False)
                .reset_index(name="averageSalePrice")
            )

            top_street = street_prices.iloc[0]
            st.metric(
                "Most expensive area in Kew",
                f"{top_street['street']}",
                help="Based on the highest average sale price for the available Kew records.",
            )

            chart_df = street_prices.head(10).set_index("street")
            st.bar_chart(chart_df)

            st.caption("Showing average sale price by street (top 10) for properties located in Kew.")


if __name__ == "__main__":
    main()
