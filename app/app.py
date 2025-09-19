import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# Ensure the project root (which contains the src package) is on the Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    PREPROCESS_DIR,
    RAW_DATA_PATTERN,
    TRAINING_DIR,
    STREET_COORDS_PATH,
)
from src.suburb_median import (
    estimate_suburb_median,
    load_suburb_median_artifacts,
)


DERIVED_NUMERIC_FEATURES = {"comparableCount"}
COMPARABLE_MATCH_FEATURES = ["street", "propertyType", "bed", "bath", "car"]


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


@st.cache_data
def load_raw_data(max_rows_per_file: int = 500) -> pd.DataFrame:
    csv_paths = sorted(DATA_DIR.glob(RAW_DATA_PATTERN))
    if not csv_paths:
        raise FileNotFoundError("No raw CSV files found in the data/ directory.")

    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        frame = pd.read_csv(path, encoding="utf-8-sig", nrows=max_rows_per_file)
        frame["__source_file"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


@st.cache_data
def load_training_sets() -> Tuple[pd.DataFrame, pd.Series]:
    X_path = TRAINING_DIR / "X.parquet"
    y_path = TRAINING_DIR / "y.parquet"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Training features not found. Run feature selection to generate X/y parquet files."
        )

    X = pd.read_parquet(X_path)
    y_df = pd.read_parquet(y_path)
    target_col = y_df.columns[0]
    return X, y_df[target_col]


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


@st.cache_resource
def load_median_resources() -> Tuple[pd.DataFrame, object, Dict[str, object]]:
    return load_suburb_median_artifacts()


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


def preview_dataframe(df: pd.DataFrame, preview_rows: int = 250) -> None:
    st.write(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    if df.empty:
        st.info("No records available for this selection.")
        return
    if df.shape[0] > preview_rows:
        st.dataframe(df.head(preview_rows))
        st.caption(f"Showing first {preview_rows} rows.")
    else:
        st.dataframe(df)


def render_inputs(
    metadata: Dict[str, object],
    median_metadata: Optional[Dict[str, object]] = None,
    show_header: bool = True,
) -> Dict[str, object]:
    categorical_features = metadata.get("categorical_features", [])
    numeric_features = metadata.get("numeric_features", [])
    categorical_levels = metadata.get("categorical_levels", {})
    numeric_summary = metadata.get("numeric_summary", {})

    inputs: Dict[str, object] = {}

    if show_header:
        st.subheader("Property Attributes")

    if categorical_features:
        st.markdown("#### Categorical")
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
        st.markdown("#### Numerical")
    for feature in numeric_features:
        if feature == "saleYear":
            stats = numeric_summary.get(feature, {})
            min_year = int(stats.get("min", 2000))
            max_year = int(stats.get("max", min_year))
            if median_metadata:
                base_year = median_metadata.get("base_year")
                max_observed = median_metadata.get("max_year")
                if isinstance(base_year, int):
                    min_year = min(min_year, base_year)
                if isinstance(max_observed, int):
                    max_year = max(max_year, max_observed)
            max_year = max(max_year, min_year)
            year_options = list(range(min_year, max_year + 2))  # allow forecasting one year ahead
            default_year = int(stats.get("median", year_options[-2] if len(year_options) > 1 else year_options[0]))
            if default_year not in year_options:
                default_year = year_options[-1]
            inputs[feature] = st.selectbox(
                feature,
                options=year_options,
                index=year_options.index(default_year),
            )
            continue
        if feature == "saleMonth":
            month_labels = [
                (1, "January"),
                (2, "February"),
                (3, "March"),
                (4, "April"),
                (5, "May"),
                (6, "June"),
                (7, "July"),
                (8, "August"),
                (9, "September"),
                (10, "October"),
                (11, "November"),
                (12, "December"),
            ]
            stats = numeric_summary.get(feature, {})
            default_month = int(round(stats.get("median", 1)))
            default_month = min(max(default_month, 1), 12)
            month_index = next((idx for idx, (value, _) in enumerate(month_labels) if value == default_month), 0)
            selection = st.selectbox(
                feature,
                options=[label for _, label in month_labels],
                index=month_index,
            )
            month_lookup = {label: value for value, label in month_labels}
            inputs[feature] = month_lookup[selection]
            continue
        if feature in DERIVED_NUMERIC_FEATURES:
            continue
        stats = numeric_summary.get(feature, {})
        min_val = float(stats.get("min", 0.0))
        max_val = float(stats.get("max", max(min_val + 1.0, 1.0)))
        median = float(stats.get("median", (min_val + max_val) / 2 if max_val > min_val else min_val))
        if max_val <= min_val:
            max_val = min_val + max(abs(min_val), 1.0)

        if feature in {"bed", "bath", "car"}:
            min_int = int(round(min_val))
            max_int = int(round(max_val)) if max_val > min_val else min_int + 5
            if max_int <= min_int:
                max_int = min_int + 5
            median_int = int(round(median))
            median_int = min(max(median_int, min_int), max_int)
            inputs[feature] = st.number_input(
                feature,
                value=median_int,
                min_value=min_int,
                max_value=max_int,
                step=1,
                format="%d",
            )
            continue

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


def _normalise_numeric(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric):
        return None
    return numeric


def derive_comparable_count(cleaned: pd.DataFrame, inputs: Dict[str, object]) -> Optional[int]:
    required_missing = [feature for feature in COMPARABLE_MATCH_FEATURES if feature not in inputs]
    if required_missing:
        return None

    bed = _normalise_numeric(inputs.get("bed"))
    bath = _normalise_numeric(inputs.get("bath"))
    car = _normalise_numeric(inputs.get("car"))
    if bed is None or bath is None or car is None:
        return None

    mask = (cleaned["street"] == inputs.get("street")) & (
        cleaned["propertyType"] == inputs.get("propertyType")
    )
    mask &= cleaned["bed"].sub(bed).abs() < 0.1
    mask &= cleaned["bath"].sub(bath).abs() < 0.1
    mask &= cleaned["car"].sub(car).abs() < 0.1

    matches = cleaned.loc[mask]
    if matches.empty:
        return None
    return int(matches["comparableCount"].iloc[0])


def render_prediction_tab(
    metadata: Dict[str, object],
    model: Optional[object],
    resource_error: Optional[str],
    median_history: Optional[pd.DataFrame],
    median_model: Optional[object],
    median_metadata: Optional[Dict[str, object]],
) -> None:
    st.subheader("Predict Next-Year Price")

    if not metadata or model is None:
        message = resource_error or (
            "Run preprocessing, feature selection, and model training to enable predictions."
        )
        st.info(message)
        return

    with st.form("prediction_form"):
        inputs = render_inputs(metadata, median_metadata)
        submitted = st.form_submit_button("Predict Price")

    if submitted:
        if median_history is None or median_model is None:
            st.warning(
                "Median baseline artefacts are missing. Run the pipeline to generate suburb medians."
            )
            return

        suburb_value = inputs.get("suburb")
        sale_year = inputs.get("saleYear")
        sale_month = inputs.get("saleMonth")

        if sale_year is None or sale_month is None:
            st.warning("Select both sale year and sale month to generate a prediction.")
            return

        sale_year_int = int(sale_year)
        sale_month_int = int(sale_month)

        baseline_details = estimate_suburb_median(
            suburb_value,
            sale_year_int,
            sale_month_int,
            median_history,
            median_model,
            median_metadata or {},
        )
        if not baseline_details or pd.isna(baseline_details.get("median")):
            st.warning(
                "Unable to determine a suburb/month baseline median for the selected inputs."
            )
            return

        baseline_value = float(baseline_details["median"])
        if baseline_value <= 0:
            st.warning("Baseline median is non-positive. Cannot produce a price estimate.")
            return

        inputs["saleYear"] = sale_year_int
        inputs["saleMonth"] = sale_month_int

        cleaned = load_cleaned_data()
        comparable_count = derive_comparable_count(cleaned, inputs)
        if comparable_count is None or comparable_count < 2:
            st.warning(
                "No comparable sales found for the selected combination. "
                "Try another street or adjust bed/bath/car values."
            )
            return

        model_inputs = dict(inputs)
        model_inputs["comparableCount"] = comparable_count

        if "baselineTransactions" in metadata.get("selected_features", []):
            transaction_count = baseline_details.get("transaction_count")
            if transaction_count is None and median_metadata:
                transaction_count = median_metadata.get("mean_transaction_count")
            if transaction_count is None:
                transaction_count = 0
            model_inputs["baselineTransactions"] = float(transaction_count)

        input_df = pd.DataFrame([model_inputs])

        expected_columns = metadata.get("selected_features", list(input_df.columns))
        missing_columns = [col for col in expected_columns if col not in input_df.columns]
        for col in missing_columns:
            input_df[col] = np.nan
        input_df = input_df[expected_columns]

        predicted_factor = float(model.predict(input_df)[0])
        predicted_price = predicted_factor * baseline_value

        st.success(f"Estimated price next year: ${predicted_price:,.0f}")
        st.caption(
            "Prediction combines the suburb/month baseline median with the property adjustment factor."
        )

        source_label = baseline_details.get("source", "observed")
        if source_label == "observed":
            baseline_caption = "Observed suburb median"
        elif source_label == "global_observed":
            baseline_caption = "Observed broader median (fallback)"
        else:
            baseline_caption = "Forecasted suburb median"

        additional = ""
        if baseline_details.get("transaction_count"):
            additional = (
                f" based on {baseline_details['transaction_count']} transactions"
            )

        st.info(f"{baseline_caption}: ${baseline_value:,.0f}{additional}")
        st.info(f"Predicted factor vs median: {predicted_factor:.3f}×")
        st.info(f"Comparable sales used: {comparable_count}")


def render_data_explorer(
    metadata: Dict[str, object],
    median_history: Optional[pd.DataFrame],
) -> None:
    st.subheader("Explore the Data Pipeline")

    stage = st.radio(
        "Dataset stage",
        (
            "Raw CSV",
            "Preprocessed",
            "Training Features",
            "Median Baselines",
            "Street Heatmap",
        ),
        horizontal=True,
        key="dataset_stage",
    )

    if stage == "Raw CSV":
        try:
            raw_df = load_raw_data()
        except FileNotFoundError as exc:
            st.info(str(exc))
            return
        preview_dataframe(raw_df)
        st.caption("Preview limited to the first 500 rows per raw file.")
        return

    if stage == "Preprocessed":
        try:
            cleaned_df = load_cleaned_data()
        except FileNotFoundError as exc:
            st.info(str(exc))
            return

        preview_dataframe(cleaned_df)

        suburb_col = "suburb" if "suburb" in cleaned_df.columns else None
        street_col = "street" if "street" in cleaned_df.columns else None
        target_col = metadata.get("target", "salePrice")

        if suburb_col and street_col and target_col in cleaned_df.columns:
            st.markdown("#### Street Highlights")
            suburbs = cleaned_df[suburb_col].dropna().unique().tolist()
            suburbs = sorted(suburbs, key=lambda value: str(value))
            if suburbs:
                selected_suburb = st.selectbox(
                    "Choose suburb",
                    options=suburbs,
                    key="suburb_selector",
                )
                suburb_df = cleaned_df[cleaned_df[suburb_col] == selected_suburb]
                if not suburb_df.empty:
                    street_summary = (
                        suburb_df.groupby(street_col, dropna=False)[target_col]
                        .mean()
                        .sort_values(ascending=False)
                        .reset_index(name="averagePrice")
                    )
                    if not street_summary.empty:
                        max_display = min(20, street_summary.shape[0])
                        top_n = min(10, max_display)
                        top_n = st.slider(
                            "Number of streets to show",
                            min_value=1,
                            max_value=max_display,
                            value=top_n,
                            key="street_limit",
                        )
                        top_subset = street_summary.head(top_n)
                        best_row = top_subset.iloc[0]
                        st.metric(
                            "Highest average street",
                            best_row[street_col],
                            f"${best_row['averagePrice']:,.0f}",
                        )
                        chart_df = top_subset.set_index(street_col)
                        st.bar_chart(chart_df["averagePrice"])
                        st.caption(
                            "Average sale price by street for the selected suburb (top results)."
                        )
        return

    if stage == "Median Baselines":
        if median_history is None:
            st.info(
                "Median baseline history not available. Run the pipeline to generate suburb/month medians."
            )
            return
        preview_dataframe(median_history)
        st.caption("Aggregated suburb × year-month medians used for baseline estimation.")
        return

    if stage == "Street Heatmap":
        try:
            coords_df = load_street_coordinates()
        except (FileNotFoundError, ValueError) as exc:
            st.info(str(exc))
            return

        try:
            cleaned_df = load_cleaned_data()
        except FileNotFoundError as exc:
            st.info(str(exc))
            return

        mitcham_df = cleaned_df[cleaned_df["suburb"].astype(str).str.upper() == "MITCHAM"].copy()
        if mitcham_df.empty:
            st.info("No Mitcham properties available in the preprocessed dataset.")
            return

        mitcham_df["saleYear"] = pd.to_numeric(mitcham_df["saleYear"], errors="coerce")
        available_years = sorted({int(year) for year in mitcham_df["saleYear"].dropna().unique()})
        if not available_years:
            st.info("Mitcham records are missing sale year information.")
            return

        default_year = available_years[-1]
        selected_year = st.slider(
            "Sale year",
            min_value=available_years[0],
            max_value=available_years[-1],
            value=default_year,
            step=1,
        )

        year_df = mitcham_df[mitcham_df["saleYear"] == selected_year]
        if year_df.empty:
            st.info(f"No Mitcham transactions recorded for {selected_year}.")
            return

        aggregated = (
            year_df.groupby("street", dropna=False)["salePrice"]
            .median()
            .reset_index(name="medianPrice")
        )
        aggregated["street"] = aggregated["street"].astype(str).str.title()

        merged = aggregated.merge(
            coords_df[coords_df["suburb"] == "MITCHAM"],
            on="street",
            how="inner",
        )

        if merged.empty:
            st.info(
                "Street coordinate mapping does not cover the Mitcham streets present in the dataset. Update config/street_coordinates.csv."
            )
            return

        merged["medianPrice"] = merged["medianPrice"].fillna(0)
        center_lat = float(merged["latitude"].mean())
        center_lng = float(merged["longitude"].mean())

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=merged,
            get_position="[longitude, latitude]",
            aggregation="MEAN",
            get_weight="medianPrice",
            radiusPixels=80,
        )

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=merged,
            get_position="[longitude, latitude]",
            get_radius="200",
            get_fill_color="[255, 140, 0, 120]",
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lng,
            zoom=13,
            pitch=40,
        )

        tooltip = {
            "html": "<b>Street:</b> {street}<br/><b>Median price:</b> ${medianPrice}",
            "style": {"backgroundColor": "#1E1E1E", "color": "#FFFFFF"},
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[heatmap_layer, scatter_layer],
                initial_view_state=view_state,
                tooltip=tooltip,
            )
        )

        st.caption(
            "Heatmap shows median sale price per street in Mitcham for the selected year."
        )
        st.dataframe(merged.sort_values("medianPrice", ascending=False))
        return

    # Training features stage
    try:
        X, y = load_training_sets()
    except FileNotFoundError as exc:
        st.info(str(exc))
        return

    target_col = metadata.get("target", y.name if hasattr(y, "name") else "target")
    combined = X.copy()
    combined[target_col] = y

    preview_dataframe(combined)
    st.caption("Feature matrix joined with the target price.")

    numeric_features = [
        col for col in metadata.get("numeric_features", []) if col in combined.columns
    ]
    if target_col not in numeric_features and target_col in combined.columns:
        numeric_features = numeric_features + [target_col]
    if numeric_features:
        st.markdown("#### Numeric Feature Summary")
        summary = combined[numeric_features].describe().transpose()
        st.dataframe(summary)

    categorical_features = [
        col for col in metadata.get("categorical_features", []) if col in combined.columns
    ]
    if categorical_features:
        st.markdown("#### Categorical Breakdown")
        selected_cat = st.selectbox(
            "Categorical feature",
            options=categorical_features,
            key="categorical_selector",
        )
        counts = combined[selected_cat].value_counts().head(15).sort_values(ascending=True)
        st.bar_chart(counts)
        st.caption("Top categories by frequency (max 15 shown).")


def render_feature_insights(
    metadata: Dict[str, object],
    model_summary: Dict[str, object],
    median_metadata: Optional[Dict[str, object]],
) -> None:
    st.subheader("Feature Insights")

    try:
        importance_df = load_feature_importances()
    except FileNotFoundError as exc:
        st.info(str(exc))
    else:
        if importance_df.empty:
            st.info("Feature importance table is empty. Rerun feature selection.")
        else:
            importance_df = importance_df.sort_values("importance", ascending=False)
            st.bar_chart(importance_df.set_index("feature").head(15))
            st.dataframe(importance_df)
            selected_features = metadata.get("selected_features", [])
            if selected_features:
                st.caption(
                    f"Selected features (total {len(selected_features)}): "
                    + ", ".join(selected_features)
                )

    st.markdown("#### Model Performance")
    try:
        metrics_df = load_model_metrics()
    except FileNotFoundError as exc:
        st.info(str(exc))
    else:
        if metrics_df.empty:
            st.info("Model metrics are unavailable. Rerun model training.")
        else:
            formatted = metrics_df.copy()
            if "mae" in formatted:
                formatted["mae"] = formatted["mae"].round(0)
            if "rmse" in formatted:
                formatted["rmse"] = formatted["rmse"].round(0)
            if "r2" in formatted:
                formatted["r2"] = formatted["r2"].round(3)
            st.dataframe(formatted)

    if model_summary:
        st.markdown("#### Selected Model")
        cols = st.columns(2)
        cols[0].metric("Best model", model_summary.get("best_model", "N/A"))
        if "r2" in model_summary:
            cols[1].metric("Validation R²", f"{model_summary['r2']:.3f}")
        best_params = model_summary.get("best_params")
        if isinstance(best_params, dict) and best_params:
            st.write("Hyper-parameters:")
            st.json(best_params)

    if median_metadata:
        st.markdown("#### Median Baseline Coverage")
        base_year = median_metadata.get("base_year")
        base_month = median_metadata.get("base_month")
        max_year = median_metadata.get("max_year")
        max_month = median_metadata.get("max_month")

        if isinstance(base_year, int) and isinstance(base_month, int):
            earliest_label = f"{base_year}-{base_month:02d}"
        else:
            earliest_label = "N/A"

        if isinstance(max_year, int) and isinstance(max_month, int):
            latest_label = f"{max_year}-{max_month:02d}"
        else:
            latest_label = "N/A"

        cols = st.columns(3)
        cols[0].metric("Earliest month", earliest_label)
        cols[1].metric("Latest observed", latest_label)
        cols[2].metric(
            "Observed months",
            str(len(median_metadata.get("observed_months", []))),
        )


def render_sidebar(model_summary: Dict[str, object]) -> None:
    st.sidebar.header("Model Summary")
    if not model_summary:
        st.sidebar.info("Run the pipeline to populate model artefacts before predicting.")
        st.sidebar.caption(
            "Ensure preprocessing and training scripts have been run before using the app."
        )
        return

    st.sidebar.write(f"Best model: {model_summary.get('best_model', 'N/A')}")
    if "r2" in model_summary:
        st.sidebar.write(f"Validation R²: {model_summary['r2']:.3f}")
    st.sidebar.write(f"Model file: {model_summary.get('model_path', 'N/A')}")

    best_params = model_summary.get("best_params")
    if isinstance(best_params, dict) and best_params:
        st.sidebar.write("Key parameters:")
        for key, value in best_params.items():
            st.sidebar.write(f"- {key}: {value}")

    st.sidebar.caption("Ensure preprocessing and training scripts have been run before using the app.")


def main() -> None:
    st.set_page_config(page_title="Property Price Predictor", layout="wide")
    st.title("Property Price Predictor")
    st.markdown(
        "Select property attributes to estimate the sale price for the coming year and explore the datasets that power the model."
    )

    metadata: Dict[str, object] = {}
    model: Optional[object] = None
    model_summary: Dict[str, object] = {}
    resource_error: Optional[str] = None
    median_history: Optional[pd.DataFrame] = None
    median_model: Optional[object] = None
    median_metadata: Dict[str, object] = {}

    try:
        metadata, model, model_summary = load_resources()
    except FileNotFoundError as exc:
        resource_error = str(exc)
        metadata_path = TRAINING_DIR / "feature_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())

    try:
        median_history, median_model, median_metadata = load_median_resources()
    except FileNotFoundError:
        pass

    if resource_error:
        st.warning(resource_error)

    render_sidebar(model_summary)

    predict_tab, explore_tab, insights_tab = st.tabs(
        ["Predict", "Explore Data", "Feature Insights"]
    )

    with predict_tab:
        render_prediction_tab(
            metadata,
            model,
            resource_error,
            median_history,
            median_model,
            median_metadata,
        )

    with explore_tab:
        render_data_explorer(metadata, median_history)

    with insights_tab:
        render_feature_insights(metadata, model_summary, median_metadata)


if __name__ == "__main__":
    main()
