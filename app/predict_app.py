"""Simple Streamlit front-end for property price prediction."""
from __future__ import annotations

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


def main() -> None:
    st.set_page_config(page_title="Property Price Predictor", layout="centered")
    st.title("Property Price Predictor")
    st.write(
        "Provide the key property characteristics below. Known streets are loaded "
        "from the latest derived dataset when available."
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
        suburb = st.selectbox("Suburb", suburbs, index=0)
    else:
        st.warning("No suburb column found in the derived dataset. Enter suburb manually.")
        suburb = st.text_input("Suburb")

    if streets:
        street = st.selectbox("Street", streets, index=0)
        st.caption(f"Street list sourced from {store_path}")
    else:
        st.warning("No street column found in the derived dataset. Enter street manually.")
        street = st.text_input("Street")

    col1, col2 = st.columns(2)
    with col1:
        sale_year = st.number_input("Sale Year", min_value=2000, max_value=2100, value=2025, step=1)
    with col2:
        sale_month = st.number_input("Sale Month", min_value=1, max_value=12, value=6, step=1)

    yearmonth = int(sale_year * 100 + sale_month)

    col3, col4, col5 = st.columns(3)
    with col3:
        bed = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
    with col4:
        bath = st.number_input("Bathrooms", min_value=0, max_value=10, value=2, step=1)
    with col5:
        car = st.number_input("Car Spaces", min_value=0, max_value=10, value=1, step=1)

    property_type = st.selectbox("Property Type", ["House", "Unit", "Apartment", "Townhouse", "Other"], index=0)

    land_size = st.number_input("Land Size (m²)", min_value=0.0, value=450.0, step=10.0)
    floor_size = st.number_input("Floor Size (m²)", min_value=0.0, value=180.0, step=10.0)
    year_built = st.number_input("Year Built", min_value=1800, max_value=int(sale_year), value=1998, step=1)

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
            st.error(f"Prediction failed: {exc}")
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
