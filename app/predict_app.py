"""Simple Streamlit front-end for property price prediction."""
from __future__ import annotations

import logging
import math
import re
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional for importance badges
    pd = None
import streamlit as st

from property_adviser.predict.feature_store import (
    feature_store_path,
    list_streets,
    list_suburbs,
    list_property_types,
    latest_sale_year_month,
)
from property_adviser.predict.model_prediction import (
    predict_property_price,
    predict_with_confidence_interval,
    load_trained_model,
    _prepare_prediction_data,
)


FEATURE_SCORE_FILES = (
    Path("data/training/feature_scores.parquet"),
    Path("data/training/feature_scores.csv"),
)

FEATURE_ALIASES: Dict[str, tuple[str, ...]] = {
    "street": ("street", "rel_street_effect"),
}


def _read_selected_features() -> list[str]:
    selected_path = Path("data/training/selected_features.txt")
    if not selected_path.exists():
        return []
    return [line.strip() for line in selected_path.read_text().splitlines() if line.strip()]


def _percentile(values: list[float], frac: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * frac
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)
    weight = k - lower
    if lower == upper:
        return values[lower]
    return values[lower] * (1 - weight) + values[upper] * weight


@lru_cache(maxsize=1)
def _feature_strengths() -> tuple[Dict[str, float], Dict[str, float]]:
    """Load feature scores once for lightweight importance indicators."""
    if pd is None:
        return {}, {}
    for path in FEATURE_SCORE_FILES:
        if not path.exists():
            continue
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        except Exception:
            continue
        if "feature" in df.columns and "best_score" in df.columns:
            grouped = df.groupby("feature")["best_score"].max().to_dict()
            break
    else:
        return {}, {}

    selected = _read_selected_features()
    selected_scores = [
        grouped[f]
        for f in selected
        if f in grouped and grouped[f] is not None and not math.isnan(grouped[f])
    ]
    if not selected_scores:
        selected_scores = [
            value
            for value in grouped.values()
            if value is not None and not math.isnan(value)
        ]

    return grouped, {
        "p25": _percentile(selected_scores, 0.25),
        "p75": _percentile(selected_scores, 0.75),
    }



@lru_cache(maxsize=1)
def _load_model_bundle() -> tuple[Any, Dict[str, Any]]:
    """Load the active model bundle exactly once per session."""
    return load_trained_model()


def _is_delta_target(metadata: Dict[str, Any]) -> bool:
    target = metadata.get("target")
    return isinstance(target, str) and target.endswith("_delta")


def _forecast_window_label(metadata: Dict[str, Any]) -> str:
    window = metadata.get("forecast_window")
    if isinstance(window, (int, float)):
        return f"{int(window)}m"
    if isinstance(window, str) and window.strip():
        return window
    target_name = metadata.get("target")
    if isinstance(target_name, str):
        match = re.search(r"(\d+)", target_name)
        if match:
            return f"{match.group(1)}m"
    return ""

def _format_strength(feature_name: str) -> str:
    score_map, thresholds = _feature_strengths()
    aliases = FEATURE_ALIASES.get(feature_name, (feature_name,))
    score = None
    for alias in aliases:
        value = score_map.get(alias)
        if value is not None:
            score = value
            break
    if score is None:
        return ""
    p25 = thresholds.get("p25", 0.25)
    p75 = thresholds.get("p75", 0.5)
    if score >= p75:
        color, label = "green", "strong"
    elif score >= p25:
        color, label = "orange", "medium"
    else:
        color, label = "blue", "light"
    return f" :{color}[{label} {score:.2f}]"


def _label(text: str, feature_name: str) -> str:
    return f"{text}{_format_strength(feature_name)}"




def main() -> None:
    st.set_page_config(page_title="Property Price Predictor", layout="centered")
    st.title("Property Price Predictor")
    st.write(
        "Estimate the expected sale price 12 months from now using the latest promoted model bundle."
    )

    score_map, thresholds = _feature_strengths()
    if score_map:
        p25 = thresholds.get("p25", 0.25)
        p75 = thresholds.get("p75", 0.5)
        st.caption(
            "Feature strength badges reflect training-time scores relative to the selected feature set: "
            f":green[strong ≥{p75:.2f}], :orange[medium ≥{p25:.2f}], :blue[light <{p25:.2f}]."
        )

    try:
        model, metadata = _load_model_bundle()
    except FileNotFoundError as exc:
        st.error(f"Model artefacts not found: {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive guard for UI
        logging.exception("prediction.load_model_failed", exc_info=exc)
        st.error("Failed to load the trained model. Please retrain or promote a model bundle.")
        return

    forecast_window = _forecast_window_label(metadata) or "12m"
    target_name = metadata.get("target", "unknown target")
    selected_model = metadata.get("selected_model") or metadata.get("model") or "unknown model"

    try:
        streets = list_streets()
        suburbs = list_suburbs()
        store_path = feature_store_path()
        observation_yearmonth = latest_sale_year_month()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    obs_year = observation_yearmonth // 100
    obs_month = observation_yearmonth % 100
    try:
        observation_label = datetime(obs_year, obs_month, 1).strftime("%B %Y")
    except ValueError:
        observation_label = str(observation_yearmonth)

    st.caption(
        f"Active model: {selected_model} → {target_name} ({forecast_window}). Baseline month: {observation_label}."
    )

    with st.expander("Debug: model inputs", expanded=False):
        st.write("Model expects these columns (in order):")
        cols = (metadata.get("feature_metadata") or {}).get("model_input_columns") or []
        st.code("\n".join(map(str, cols)) or "(none)")
        if st.button("Show prepared features for current form", key="show_prepared"):
            try:
                props = [{
                    "yearmonth": observation_yearmonth,
                    "bed": 3,
                    "bath": 2,
                    "car": 1,
                    "propertyType": "House",
                    "street": streets[0] if streets else "",
                    "suburb": suburbs[0] if suburbs else "",
                    "landSize": 450.0,
                    "floorSize": 180.0,
                    "yearBuild": 1998,
                }]
                prepared = _prepare_prediction_data(props, metadata)
                st.dataframe(prepared)
            except Exception as exc:  # pragma: no cover
                st.warning(f"Failed to build prepared features: {exc}")

    suburb = None
    if suburbs:
        suburb = st.selectbox(_label("Suburb", "suburb"), suburbs, index=0)
    else:
        st.warning("No suburb column found in the derived dataset. Enter suburb manually.")
        suburb = st.text_input(_label("Suburb", "suburb"))

    if streets:
        street = st.selectbox(_label("Street", "street"), streets, index=0)
        st.caption(f"Street list sourced from {store_path}")
    else:
        st.warning("No street column found in the derived dataset. Enter street manually.")
        street = st.text_input(_label("Street", "street"))

    col1, col2, col3 = st.columns(3)
    with col1:
        bed = st.number_input(
            _label("Bedrooms", "bed"),
            min_value=0,
            max_value=10,
            value=3,
            step=1,
        )
    with col2:
        bath = st.number_input(
            _label("Bathrooms", "bath"),
            min_value=0,
            max_value=10,
            value=2,
            step=1,
        )
    with col3:
        car = st.number_input(
            _label("Car Spaces", "car"),
            min_value=0,
            max_value=10,
            value=1,
            step=1,
        )

    property_types = list_property_types()
    if not property_types:
        property_types = ["House", "Unit", "Apartment"]
    property_type = st.selectbox(
        _label("Property Type", "propertyType"),
        property_types,
        index=0,
    )

    land_size = st.number_input(
        _label("Land Size (m²)", "landSizeM2"),
        min_value=0.0,
        value=450.0,
        step=10.0,
    )
    floor_size = st.number_input(
        _label("Floor Size (m²)", "floorSizeM2"),
        min_value=0.0,
        value=180.0,
        step=10.0,
    )

    current_year = datetime.now().year
    default_year_built = min(current_year, 1998)
    year_built = st.number_input(
        _label("Year Built", "yearBuilt"),
        min_value=1800,
        max_value=current_year,
        value=default_year_built,
        step=1,
    )

    button_label = f"Predict {forecast_window} price"
    if st.button(button_label, type="primary"):
        if not street:
            st.error("Street is required for prediction.")
            return
        if not suburb:
            st.error("Suburb is required for prediction.")
            return
        try:
            result = predict_with_confidence_interval(
                yearmonth=observation_yearmonth,
                bed=int(bed),
                bath=int(bath),
                car=int(car),
                property_type=property_type,
                street=street,
                suburb=suburb,
                land_size=land_size or None,
                floor_size=floor_size or None,
                year_built=int(year_built) if year_built else None,
                model=model,
                metadata=metadata,
            )
            price = result["predicted_price"]
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - surfacing to UI
            logging.exception("prediction.failed", exc_info=exc)
            st.error(f"Prediction failed: {exc}")
            st.code(traceback.format_exc())
            return

        st.subheader(f"Projected Price ({forecast_window} outlook)")
        st.metric("Predicted Price", f"${price:,.0f}")

        raw_delta = result.get("raw_prediction")
        base_value = result.get("base_value")
        if raw_delta is not None and _is_delta_target(metadata):
            delta_pct = raw_delta * 100.0
            if base_value:
                st.caption(
                    f"Projected change: {delta_pct:+.2f}% relative to ${base_value:,.0f}."
                )
            else:
                st.caption(f"Projected change: {delta_pct:+.2f}%")

        margin = result["upper_bound"] - result["lower_bound"]
        st.caption(
            "Confidence interval assumes normal error using validation RMSE. "
            f"(±${margin / 2:,.0f} around the estimate at {result['confidence_level']:.0%} confidence.)"
        )
        st.write(
            f"Lower bound: ${result['lower_bound']:,.0f} — Upper bound: ${result['upper_bound']:,.0f}"
        )




if __name__ == "__main__":
    main()
