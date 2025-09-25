# preprocess_derive.py

from typing import Any, Dict
import numpy as np
import pandas as pd
import re

from src.common.app_logging import log, warn
from src.common.runner import run_step

UNIT_KEYWORDS = ("unit", "flat", "apartment", "villa", "studio", "duplex")
HOUSE_KEYWORDS = ("house", "detached", "dwelling", "residence", "townhouse")


# --- Feature Extraction ---
def extract_street(address: str, cfg: Dict[str, Any]) -> str:
    """Extract and clean street name from a full address string."""
    unknown = cfg.get("unknown_value", "Unknown")
    if not isinstance(address, str):
        return unknown
    # Remove unit/house number prefixes
    cleaned = re.sub(r"^[0-9]+[A-Z]?/[0-9]+\s+", "", address)
    cleaned = re.sub(r"^[0-9]+[A-Z]?\s+", "", address).strip()

    return cleaned.title() if cleaned else unknown

def simplify_property_type(raw: Any, cfg: Dict[str, Any]) -> str:
    """Standardise property type (e.g. Unit vs House) from raw string."""
    unknown = cfg.get("unknown_value", "")
    if not isinstance(raw, str):
        return unknown

    for k in cfg.get("unit_keywords", UNIT_KEYWORDS):
        if k in raw:
            return cfg.get("unit_label", "Unit")
    for k in cfg.get("house_keywords", HOUSE_KEYWORDS):
        if k in raw:
            return cfg.get("house_label", "House")
    return unknown

# --- Derived Features ---

def apply_group_aggregate(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """Aggregate a column by groups and merge result back into original dataframe."""
    if not spec.get("enabled", True):
        return df

    group_by, target, agg, output = spec["group_by"], spec["target"], spec["aggregate"], spec["output"]
    if any(col not in df.columns for col in group_by + [target]) or output in df.columns:
        return df

    aggregated = df[group_by + [target]].groupby(group_by, dropna=False)[target].agg(agg).reset_index(name=output)
    return df.merge(aggregated, on=group_by, how="left")

def derive_price_factor(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """Calculate a price factor by dividing two columns and filter out invalid values."""
    if not spec.get("enabled", True):
        return df

    num, den_col, out = spec["numerator"], spec["denominator"], spec["output"]
    min_val = float(spec.get("min_value", 0.01))
    missing_cols = [col for col in [num, den_col] if col not in df.columns]

    if missing_cols:
        warn("derive.price_factor", reason="missing_columns", missing=missing_cols)
        return df

    before = df.shape[0]
    df[out] = df[num] / df[den_col]
    df[out] = df[out].replace([np.inf, -np.inf], np.nan)
    df = df[df[out].notna() & (df[out] > min_val)].copy()

    log("derive.price_factor", numerator=num, denominator=den_col, output=out,
        min_value=min_val, rows_in=before, rows_out=df.shape[0])
    return df

def _derive_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year/month components from a 'saleDate' column."""
    if "saleDate" not in df.columns:
        return df

    sd = pd.to_datetime(df["saleDate"], errors="coerce")
    if sd.notna().any():
        df["saleYear"] = sd.dt.year
        df["saleMonth"] = sd.dt.month
        df["saleYearMonth"] = sd.dt.year * 100 + sd.dt.month
        log("derive.date_parts", cols=["saleYear", "saleMonth", "saleYearMonth"])
    return df

# --- Main Entry ---
def derive_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Run configured derivation steps on the dataframe."""
    df = _derive_date_parts(df)
    derivations = config["derivations"]

    for name, spec in derivations.items():
        if not spec.get("enabled", True):
            log("derive.step", status="skipped", name=name, reason="disabled")
            continue

        if name == "street":
            src, out = spec["source"], spec["output"]
            drop_unknown = spec.get("drop_unknown", False)
            cfg = spec.get("config", {"unknown_value": "Unknown"})

            def _street(dfin):
                if src in dfin.columns:
                    before = dfin.shape[0]
                    dfin[out] = dfin[src].apply(lambda x: extract_street(x, cfg))
                    if drop_unknown:
                        dfin = dfin[dfin[out] != cfg["unknown_value"]].copy()
                    log("derive.street", source=src, output=out, drop_unknown=drop_unknown,
                        rows_in=before, rows_out=dfin.shape[0])
                else:
                    warn("derive.street", reason="missing_source", source=src)
                return dfin

            df = run_step("derive.street", _street, df)
            continue

        if name == "propertyType":
            cfg = spec.get("config", {"unknown_value": ""})

            def _ptype(dfin):
                if "propertyType" not in dfin.columns:
                    warn("derive.property_type", reason="missing_column", column="propertyType")
                    return dfin
                before = dfin.shape[0]
                dfin["propertyType"] = dfin["propertyType"].apply(lambda x: simplify_property_type(x, cfg))
                if cfg["unknown_value"]:
                    dfin = dfin[dfin["propertyType"] != cfg["unknown_value"]].copy()
                log("derive.property_type", rows_in=before, rows_out=dfin.shape[0])
                return dfin

            df = run_step("derive.property_type", _ptype, df)
            continue

        if all(k in spec for k in ("group_by", "target", "aggregate", "output")):
            df = run_step("derive.group_aggregate", apply_group_aggregate, df, spec=spec)
            continue

        if name == "postcode_prefix":
            src, out, length = spec["source"], spec["output"], int(spec["length"])

            def _postcode(dfin):
                if src not in dfin.columns or out in dfin.columns:
                    return dfin
                dfin[out] = dfin[src].apply(lambda v: str(v).strip()[:length] if pd.notna(v) else None)
                return dfin

            df = run_step("derive.postcode_prefix", _postcode, df)
            continue

        if name == "price_factor":
            df = run_step("derive.price_factor", derive_price_factor, df, spec=spec)
            continue

        warn("derive.step", name=name, reason="unknown_step")

    return df