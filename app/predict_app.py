"""Simple Streamlit front-end for property price prediction."""
from __future__ import annotations

import logging
import traceback
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional for importance badges
    pd = None
import streamlit as st

from property_adviser.predict.feature_store import (
    feature_store_path,
    list_streets,
    list_suburbs,
)
from property_adviser.predict.model_prediction import (
    predict_property_price,
    predict_with_confidence_interval,
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
        "Provide the key property characteristics below. Known streets are loaded "
        "from the latest derived dataset when available."
    )
    score_map, thresholds = _feature_strengths()
    if score_map:
        p25 = thresholds.get("p25", 0.25)
        p75 = thresholds.get("p75", 0.5)
        st.caption(
            "Feature strength badges reflect training-time scores relative to the selected "
            f"feature set: :green[strong ≥{p75:.2f}], :orange[medium ≥{p25:.2f}], "
            f":blue[light <{p25:.2f}]."
        )

    try:
        streets = list_streets()
        suburbs = list_suburbs()
        store_path = feature_store_path()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

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

    col1, col2 = st.columns(2)
    with col1:
        sale_year = st.number_input(
            _label("Sale Year", "saleYear"),
            min_value=2000,
            max_value=2100,
            value=2025,
            step=1,
        )
    with col2:
        sale_month = st.number_input(
            _label("Sale Month", "saleMonth"),
            min_value=1,
            max_value=12,
            value=6,
            step=1,
        )

    yearmonth = int(sale_year * 100 + sale_month)

    col3, col4, col5 = st.columns(3)
    with col3:
        bed = st.number_input(
            _label("Bedrooms", "bed"),
            min_value=0,
            max_value=10,
            value=3,
            step=1,
        )
    with col4:
        bath = st.number_input(
            _label("Bathrooms", "bath"),
            min_value=0,
            max_value=10,
            value=2,
            step=1,
        )
    with col5:
        car = st.number_input(
            _label("Car Spaces", "car"),
            min_value=0,
            max_value=10,
            value=1,
            step=1,
        )

    property_type = st.selectbox(
        _label("Property Type", "propertyType"),
        ["House", "Unit", "Apartment", "Townhouse", "Other"],
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
    year_built = st.number_input(
        _label("Year Built", "yearBuilt"),
        min_value=1800,
        max_value=int(sale_year),
        value=1998,
        step=1,
    )

    if st.button("Predict Price", type="primary"):
        if not street:
            st.error("Street is required for prediction.")
            return
        if not suburb:
            st.error("Suburb is required for prediction.")
            return
        try:
            price = predict_property_price(
                yearmonth=yearmonth,
                bed=int(bed),
                bath=int(bath),
                car=int(car),
                property_type=property_type,
                street=street,
                suburb=suburb,
                land_size=land_size or None,
                floor_size=floor_size or None,
                year_built=int(year_built) if year_built else None,
            )
            ci = predict_with_confidence_interval(
                yearmonth=yearmonth,
                bed=int(bed),
                bath=int(bath),
                car=int(car),
                property_type=property_type,
                street=street,
                suburb=suburb,
                land_size=land_size or None,
                floor_size=floor_size or None,
                year_built=int(year_built) if year_built else None,
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - surfacing to UI
            logging.exception("prediction.failed", exc_info=exc)
            st.error(f"Prediction failed: {exc}")
            st.code(traceback.format_exc())
            return

        st.subheader("Estimate")
        st.metric("Predicted Price", f"${price:,.0f}")

        margin = ci["upper_bound"] - ci["lower_bound"]
        st.caption(
            "Confidence interval assumes normal error using validation RMSE. "
            f"(±${margin / 2:,.0f} around the estimate at {ci['confidence_level']:.0%} confidence.)"
        )
        st.write(
            f"Lower bound: ${ci['lower_bound']:,.0f} — Upper bound: ${ci['upper_bound']:,.0f}"
        )


if __name__ == "__main__":
    main()
