import pandas as pd
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from property_adviser.common.app_utils import (
    load_cleaned_data
)

st.set_page_config(page_title="Property Price Overview", layout="wide")
st.title("Property Price Overview")
st.caption(
    "Visualise suburb street-level pricing, explore correlations, and monitor model performance."
)


try:
    cleaned = load_cleaned_data()
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.stop()

suburb_column = "suburb" if "suburb" in cleaned.columns else None
street_column = "street" if "street" in cleaned.columns else None
price_column = "salePrice" if "salePrice" in cleaned.columns else None
year_column = "saleYear" if "saleYear" in cleaned.columns else None

if not suburb_column or not street_column or not price_column:
    st.warning(
        "Cleaned dataset is missing suburb/street/salePrice columns required for the overview map."
    )
    st.stop()

cleaned[suburb_column] = cleaned[suburb_column].astype(str).str.strip()
suburb_values = (
    cleaned[suburb_column]
    .dropna()
    .str.upper()
    .sort_values()
    .unique()
    .tolist()
)

if not suburb_values:
    st.warning("No suburbs available in the cleaned dataset.")
    st.stop()

# --- ADD THIS CODE SNIPPET ---
# Ensure the price column is a numeric type before any calculations
if pd.api.types.is_object_dtype(cleaned[price_column]):
    # Clean the column: remove commas and dollar signs
    cleaned[price_column] = cleaned[price_column].astype(str).str.replace('[\$,]', '', regex=True)
    # Convert to numeric, coercing any errors to NaN
    cleaned[price_column] = pd.to_numeric(cleaned[price_column], errors='coerce')
# --- END OF SNIPPET ---

selected_suburb_label = st.selectbox(
    "Suburb",
    options=suburb_values,
    format_func=lambda value: value.title(),
)

subset = cleaned[cleaned[suburb_column].str.upper() == selected_suburb_label].copy()

if subset.empty:
    st.warning(f"No records available for suburb {selected_suburb_label.title()}.")
    st.stop()

if year_column and subset[year_column].notna().any():
    subset[year_column] = pd.to_numeric(subset[year_column], errors="coerce")
    years = sorted(subset[year_column].dropna().unique())
    if years:
        min_year = int(min(years))
        max_year = int(max(years))
        start, end = st.slider(
            "Sale year range",
            min_value=min_year,
            max_value=max_year,
            value=(max(min_year, max_year - 1), max_year),
        )
        subset = subset[(subset[year_column] >= start) & (subset[year_column] <= end)]
        st.caption(f"Showing results for {start} to {end}.")

if subset.empty:
    st.info("No transactions match the current filters.")
    st.stop()

filter_info = {}

# Create filter row with enough columns for all filters
filter_row = st.columns(6)  # Increased from 5 to 6 columns to accommodate all filters
filter_idx = 0

numeric_filters = [
    ("bed", "Bedrooms"),
    ("bath", "Bathrooms"),
    ("car", "Car spaces"),
]
for idx, (column, label) in enumerate(numeric_filters):
    if column in subset.columns and pd.api.types.is_numeric_dtype(subset[column]):
        series = subset[column].dropna()
        if series.empty:
            continue
        options = sorted(series.unique())
        options = [option for option in options if pd.notna(option)]
        if not options:
            continue
        options_display = [int(option) if float(option).is_integer() else float(option) for option in options]
        selection = filter_row[filter_idx].selectbox(
            label,
            options=["All"] + options_display,
            index=0,
        )
        if selection != "All":
            subset = subset[subset[column] == selection]
            filter_info[label] = [selection]
        filter_idx += 1

# Property type filter
property_column = "propertyType"
if property_column in subset.columns:
    options = (
        subset[property_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    options = sorted(options)
    if options:
        selection = filter_row[filter_idx].selectbox(
            "Property type",
            options=["All"] + options,
            index=0,
        )
        if selection != "All":
            subset = subset[subset[property_column] == selection]
            filter_info["Property type"] = [selection]
        filter_idx += 1

# Street filter
street_filter = filter_row[filter_idx].text_input("Street contains", "").strip()
if street_filter:
    subset = subset[subset[street_column].astype(str).str.contains(street_filter, case=False, na=False)]
    filter_info["Street"] = [street_filter]

filtered_empty = subset.empty
if filtered_empty:
    st.info("Filters removed all rows for the selected suburb.")

street_summary = pd.DataFrame()
if not filtered_empty:
    streetly = subset.copy()
    streetly[street_column] = streetly[street_column].astype(str).str.title()
    street_summary = (
        streetly.groupby(street_column, dropna=False)[price_column]
        .agg(medianPrice="median", transactions="size")
        .reset_index()
        .rename(columns={street_column: "street"})
    )
    street_summary = street_summary.sort_values("medianPrice", ascending=False)
    street_summary["medianPriceDisplay"] = street_summary["medianPrice"].apply(
        lambda value: f"${value:,.0f}"
    )

if street_summary.empty:
    st.info(
        "No street-level aggregates available. TODO: restore map visualisation once coordinates are supplied."
    )
else:
    best_street = street_summary.iloc[0]
    st.markdown("### Highest median-priced street")
    st.metric(
        label=best_street["street"],
        value=best_street["medianPriceDisplay"],
        delta=f"Transactions: {int(best_street['transactions'])}"
    )

    st.markdown("### Top streets by median price")
    top10 = street_summary[["street", "medianPriceDisplay", "transactions"]].head(10)
    top10 = top10.rename(
        columns={
            "street": "Street",
            "medianPriceDisplay": "Median price",
            "transactions": "Transactions",
        }
    )
    st.dataframe(top10, hide_index=True)

row_count = subset.shape[0]
st.markdown(f"**Selected rows:** {row_count:,}")

if filter_info:
    parts = []
    for label, value in filter_info.items():
        parts.append(f"{label}: {', '.join(map(str, value))}")
    if parts:
        st.caption("Active filters: " + ", ".join(parts))

st.markdown("### Filtered Results")
st.dataframe(subset)