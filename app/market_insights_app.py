"""Streamlit dashboard for analysing market drivers, demand dynamics, and price timelines."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from property_adviser.predict.feature_store import feature_store_path
from property_adviser.core.config import load_config

FEATURE_SCORES_CANDIDATES = [
    Path("data/training/feature_scores.parquet"),
    Path("data/training/feature_scores.csv"),
]
FEATURE_CONFIG_PATH = Path("config/features.yml")


def _load_derived() -> pd.DataFrame:
    path = feature_store_path()
    if not path.exists():
        raise FileNotFoundError(
            "Derived dataset not found. Run preprocessing to generate data/preprocess/derived.*"
        )
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df["saleYearMonth"] = pd.to_numeric(df["saleYearMonth"], errors="coerce")
    df["propertyType"] = df["propertyType"].fillna("Unknown")
    df["sale_period"] = pd.to_datetime(
        df["saleYearMonth"].dropna().astype(int).astype(str) + "01",
        format="%Y%m%d",
        errors="coerce",
    )
    df = df[df["sale_period"].notna()].copy()
    return df


@st.cache_data(show_spinner=False)
def load_derived_cached() -> pd.DataFrame:
    return _load_derived()


@st.cache_data(show_spinner=False)
def load_feature_scores_cached() -> Optional[pd.DataFrame]:
    for candidate in FEATURE_SCORES_CANDIDATES:
        if candidate.exists():
            if candidate.suffix == ".parquet":
                return pd.read_parquet(candidate)
            return pd.read_csv(candidate)
    return None


@lru_cache(maxsize=1)
def feature_themes() -> dict[str, str]:
    return {
        "bed": "Bedrooms",
        "bath": "Bathrooms",
        "car": "Car Spaces",
        "landSizeM2": "Land Size",
        "floorSizeM2": "Floor Size",
        "propertyType": "Property Type",
        "propertyAge": "Age / Vintage",
        "propertyAgeBand": "Age / Vintage",
        "street": "Street Effect",
        "suburb_price_median_current": "Market Context",
        "suburb_price_median_3m": "Market Context",
        "suburb_price_median_6m": "Market Context",
        "suburb_price_median_12m": "Market Context",
        "suburb_txn_count_3m": "Liquidity",
        "suburb_txn_count_6m": "Liquidity",
        "suburb_txn_count_12m": "Liquidity",
        "suburb_delta_3m": "Momentum",
        "suburb_delta_12m": "Momentum",
        "rel_price_vs_suburb_median": "Relative Pricing",
    }


@lru_cache(maxsize=1)
def excluded_features() -> set[str]:
    try:
        cfg = load_config(FEATURE_CONFIG_PATH)
    except Exception:
        return set()
    cols = cfg.get("exclude_columns", [])
    return set(cols) if isinstance(cols, (list, tuple, set)) else set()


def prepare_feature_scores(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    if "best_score" not in df.columns:
        metric_cols = [col for col in ["pearson_abs", "mutual_info", "eta"] if col in df.columns]
        df["best_score"] = df[metric_cols].max(axis=1, skipna=True)
        df["best_metric"] = df[metric_cols].idxmax(axis=1, skipna=True)
    df = df.sort_values("best_score", ascending=False)
    buckets = feature_themes()
    df["driver_theme"] = df["feature"].map(lambda f: buckets.get(f, "Other / Engineered"))
    exclude = excluded_features()
    if exclude:
        df = df[~df["feature"].isin(exclude)]
    return df


def driver_tab(df: pd.DataFrame, scores_df: Optional[pd.DataFrame]) -> None:
    st.header("Market Drivers")
    st.write("See which features matter most to the model and how they influence price.")

    if scores_df is None or scores_df.empty:
        st.warning("Feature scores not found. Run feature selection to populate them.")
        return

    filtered_scores = scores_df.copy()
    theme_options = ["All Themes"] + sorted(filtered_scores["driver_theme"].unique())
    chosen_theme = st.selectbox("Filter by driver theme", theme_options, index=0)
    if chosen_theme != "All Themes":
        filtered_scores = filtered_scores[filtered_scores["driver_theme"] == chosen_theme]

    top_n = st.slider("Display top N drivers", min_value=5, max_value=30, value=10, step=1)
    top_scores = filtered_scores.head(top_n)

    chart = (
        alt.Chart(top_scores)
        .mark_bar()
        .encode(
            x=alt.X("best_score:Q", title="Importance Score"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.Color("driver_theme:N", legend=alt.Legend(title="Theme")),
            tooltip=["feature", "best_score", "best_metric", "driver_theme"],
        )
        .properties(height=alt.Step(18), width="container")
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Driver vs Sale Price")
    exclude = excluded_features()
    numeric_cols = sorted(
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"salePrice", "saleMonth", "saleYearMonth"} and col not in exclude
    )
    if not numeric_cols:
        st.info("No numeric columns available for scatter analysis.")
        return

    chosen_feature = st.selectbox("Choose a numeric driver", numeric_cols, index=0)
    years = df["saleYear"].dropna().astype(int)
    if years.empty:
        year_filtered = df
        year_range = None
    else:
        min_year, max_year = int(years.min()), int(years.max())
        default_start = max(min_year, max_year - 4)
        year_range = st.slider(
            "Sale year range",
            min_year,
            max_year,
            (default_start, max_year),
            step=1,
        )
        mask = df["saleYear"].between(year_range[0], year_range[1])
        year_filtered = df[mask]

    sample_df = year_filtered[[chosen_feature, "salePrice", "saleYear"]].dropna().copy()
    if sample_df.empty:
        st.info("Not enough data for the chosen driver.")
        return

    scatter = (
        alt.Chart(sample_df.sample(min(len(sample_df), 4000), random_state=7))
        .mark_circle(size=45, opacity=0.45)
        .encode(
            x=alt.X(f"{chosen_feature}:Q"),
            y=alt.Y("salePrice:Q", title="Sale Price"),
            color=alt.Color("saleYear:O", title="Sale Year"),
            tooltip=[chosen_feature, "salePrice", "saleYear"],
        )
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)

    st.caption(
        "Colour encodes sale year for the selected range. Use the slider to focus on specific periods."
    )

    st.subheader("Average price by driver bin and year")
    binned_chart = (
        alt.Chart(sample_df)
        .transform_bin("driver_bin", chosen_feature, maxbins=20)
        .mark_line(point=True)
        .encode(
            x=alt.X("driver_bin:Q", title=f"{chosen_feature} (binned)"),
            y=alt.Y("mean(salePrice):Q", title="Mean Sale Price"),
            color=alt.Color("saleYear:O", title="Sale Year"),
            tooltip=["saleYear", "driver_bin", alt.Tooltip("mean(salePrice):Q", title="Mean Price", format=",.0f")],
        )
        .properties(height=300)
    )
    st.altair_chart(binned_chart, use_container_width=True)


def demand_tab(df: pd.DataFrame) -> None:
    st.header("Demand & Growth Explorer")
    suburbs = sorted(df["suburb"].dropna().unique())
    selected_suburbs = st.multiselect(
        "Select suburbs to focus on (leave empty for all)", suburbs, default=[]
    )
    filtered = df[df["suburb"].isin(selected_suburbs)] if selected_suburbs else df

    growth = (
        filtered.groupby(["propertyType", "sale_period"])
        .agg(median_price=("salePrice", "median"), txn_count=("salePrice", "size"))
        .reset_index()
    )
    if growth.empty:
        st.info("No transactions available with the current filters.")
        return

    growth["median_price_12m_ago"] = (
        growth.sort_values("sale_period")
        .groupby("propertyType")["median_price"]
        .shift(12)
    )
    growth["yoy_growth"] = (
        growth["median_price"] - growth["median_price_12m_ago"]
    ) / growth["median_price_12m_ago"]

    yoy_summary = (
        growth.dropna(subset=["yoy_growth"])
        .groupby("propertyType")
        .agg(
            latest_growth=("yoy_growth", "last"),
            latest_price=("median_price", "last"),
            txn_last12m=("txn_count", "sum"),
        )
        .sort_values("latest_growth", ascending=False)
    )
    yoy_summary["latest_growth_pct"] = yoy_summary["latest_growth"].apply(
        lambda v: f"{v*100:,.1f}%" if pd.notna(v) else "N/A"
    )
    st.dataframe(
        yoy_summary[["latest_price", "txn_last12m", "latest_growth_pct"]],
        use_container_width=True,
    )

    growth_chart = (
        alt.Chart(growth)
        .mark_line()
        .encode(
            x=alt.X("sale_period:T", title="Sale Period"),
            y=alt.Y("median_price:Q", title="Median Sale Price"),
            color="propertyType",
            tooltip=["propertyType", "sale_period", "median_price", "txn_count", "yoy_growth"],
        )
        .interactive()
    )
    st.altair_chart(growth_chart, use_container_width=True)

    st.subheader("Demand Heatmap (Volume Share)")
    demand = (
        filtered.groupby(["suburb", "propertyType"])
        .agg(txn_count=("salePrice", "size"))
        .reset_index()
    )
    total_by_suburb = demand.groupby("suburb")["txn_count"].transform("sum")
    demand["volume_share"] = demand["txn_count"] / total_by_suburb

    heatmap = (
        alt.Chart(demand)
        .mark_rect()
        .encode(
            x=alt.X("propertyType:N"),
            y=alt.Y("suburb:N"),
            color=alt.Color("volume_share:Q", title="Volume Share"),
            tooltip=["suburb", "propertyType", "txn_count", alt.Tooltip("volume_share:Q", format=".1%")],
        )
    )
    st.altair_chart(heatmap, use_container_width=True)


def timeline_tab(df: pd.DataFrame) -> None:
    st.header("Price Timeline Studio")
    default_suburb = df["suburb"].mode().iat[0] if not df["suburb"].empty else None
    suburbs = sorted(df["suburb"].unique())
    suburb_index = suburbs.index(default_suburb) if default_suburb in suburbs else 0
    suburb = st.selectbox("Choose a suburb", suburbs, index=suburb_index)

    property_types = sorted(df["propertyType"].unique())
    selected_types = st.multiselect(
        "Property types to plot", property_types, default=property_types
    )

    subset = df[(df["suburb"] == suburb) & (df["propertyType"].isin(selected_types))]
    if subset.empty:
        st.info("No data for the chosen suburb/property types.")
        return

    timeline = (
        subset.groupby(["propertyType", "sale_period"])
        .agg(
            median_price=("salePrice", "median"),
            txn_count=("salePrice", "size"),
            volatility=("salePrice", "std"),
        )
        .reset_index()
    )

    price_chart = (
        alt.Chart(timeline)
        .mark_line(point=True)
        .encode(
            x=alt.X("sale_period:T", title="Sale Period"),
            y=alt.Y("median_price:Q", title="Median Sale Price"),
            color="propertyType",
            tooltip=["propertyType", "sale_period", "median_price", "txn_count"],
        )
        .interactive()
    )
    st.altair_chart(price_chart, use_container_width=True)

    st.caption(
        "Hover to inspect monthly medians, transaction counts, and compare property types."
    )

    volatility_chart = (
        alt.Chart(timeline)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("sale_period:T"),
            y=alt.Y("txn_count:Q", title="Transactions"),
            color="propertyType",
            tooltip=["propertyType", "sale_period", "txn_count", "volatility"],
        )
        .properties(height=200)
    )
    st.altair_chart(volatility_chart, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Market Insights", layout="wide")
    st.title("Market Insights Dashboard")
    st.write(
        "Explore the drivers behind sale prices, discover where demand is shifting, and visualise price trajectories."
    )

    try:
        derived_df = load_derived_cached()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    feature_scores = load_feature_scores_cached()
    if feature_scores is not None:
        feature_scores = prepare_feature_scores(feature_scores)

    tab1, tab2, tab3 = st.tabs(
        ["Market Drivers", "Demand & Growth", "Price Timelines"]
    )

    with tab1:
        driver_tab(derived_df, feature_scores)

    with tab2:
        demand_tab(derived_df)

    with tab3:
        timeline_tab(derived_df)


if __name__ == "__main__":
    main()
