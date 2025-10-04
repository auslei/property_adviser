"""Utilities to hydrate prediction features from the derived dataset."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from property_adviser.config import PREPROCESS_CONFIG_PATH, PREPROCESS_DIR, PROJECT_ROOT
from property_adviser.core.config import load_config


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@lru_cache(maxsize=1)
def _derive_output_path() -> Path:
    try:
        pre_cfg = load_config(PREPROCESS_CONFIG_PATH)
        derivation_cfg = pre_cfg.get("derivation", {})
        output = derivation_cfg.get("output_path")
        if output:
            candidate = _resolve_path(output)
            if candidate.exists():
                return candidate
    except Exception:
        pass

    # fallbacks
    for candidate in (
        PREPROCESS_DIR / "derived.parquet",
        PREPROCESS_DIR / "derived.csv",
    ):
        if candidate.exists():
            return candidate
    return PREPROCESS_DIR / "derived.parquet"


@lru_cache(maxsize=1)
def _derive_detail_path() -> Optional[Path]:
    try:
        pre_cfg = load_config(PREPROCESS_CONFIG_PATH)
        derivation_cfg = pre_cfg.get("derivation", {})
        detail_path = derivation_cfg.get("detailed_output_path")
        if detail_path:
            candidate = _resolve_path(detail_path)
            if candidate.exists():
                return candidate
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _street_column_name() -> str:
    try:
        pre_cfg = load_config(PREPROCESS_CONFIG_PATH)
        derive_path = pre_cfg.get("derivation", {}).get("config_path")
        if derive_path:
            derive_cfg = load_config(_resolve_path(derive_path))
            street_cfg = derive_cfg.get("street", {})
            return street_cfg.get("output", "street")
    except Exception:
        pass
    return "street"


@lru_cache(maxsize=1)
def _load_dataframe() -> pd.DataFrame:
    path = _derive_output_path()
    if not path.exists():
        raise FileNotFoundError(
            "Derived dataset not found. Run preprocessing to generate it before prediction."
        )
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_detail_dataframe() -> pd.DataFrame:
    detail_path = _derive_detail_path()
    if detail_path is None or not detail_path.exists():
        raise FileNotFoundError
    if detail_path.suffix.lower() == ".parquet":
        return pd.read_parquet(detail_path)
    return pd.read_csv(detail_path)


@lru_cache(maxsize=1)
def latest_sale_year_month() -> int:
    """Return the most recent saleYearMonth (or observingYearMonth) present in the derived dataset."""
    df = _load_dataframe()
    for column in ("saleYearMonth", "observingYearMonth", "observing_year_month"):
        if column in df.columns:
            series = pd.to_numeric(df[column], errors='coerce').dropna()
            if not series.empty:
                return int(series.max())
    raise ValueError("Derived dataset is missing a usable sale year/month column.")


def feature_store_path() -> Path:
    path = _derive_output_path()
    if path.exists():
        return path
    # Trigger loader to raise a consistent error
    _load_dataframe()
    return path


def list_streets() -> List[str]:
    try:
        df = _load_detail_dataframe()
    except FileNotFoundError:
        df = _load_dataframe()
    street_col = _street_column_name()
    if street_col not in df.columns:
        return []
    streets = (
        df[street_col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s.str.len() > 0]
        .loc[lambda s: s.str.lower() != "unknown"]
        .loc[lambda s: ~s.str.contains(r"\d")]
        .map(str.title)
        .unique()
    )
    return sorted(streets.tolist())


def list_suburbs() -> List[str]:
    df = _load_dataframe()
    if "suburb" not in df.columns:
        return []
    suburbs = (
        df["suburb"].dropna().astype(str).map(str.strip).loc[lambda s: s.str.len() > 0]
    )
    suburbs = suburbs.map(str.title).unique()
    return sorted(suburbs.tolist())


def fetch_reference_features(
    suburb: str,
    sale_year_month: int,
    columns: Iterable[str],
) -> pd.Series:
    """Return representative feature values for a suburb/month combination."""
    df = _load_dataframe()
    if "suburb" not in df.columns:
        return pd.Series(index=list(columns), dtype=object)

    suburb_normalised = suburb.strip().upper()
    subset = df[df["suburb"].astype(str).str.upper() == suburb_normalised]
    if subset.empty:
        return pd.Series(index=list(columns), dtype=object)

    if "saleYearMonth" in subset.columns:
        subset = subset.sort_values("saleYearMonth")
        exact = subset[subset["saleYearMonth"] == sale_year_month]
        if not exact.empty:
            row = exact.iloc[-1]
        else:
            prior = subset[subset["saleYearMonth"] <= sale_year_month]
            row = prior.iloc[-1] if not prior.empty else subset.iloc[-1]
    else:
        row = subset.iloc[-1]

    return row.reindex(index=list(columns))
