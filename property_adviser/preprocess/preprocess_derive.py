# preprocess_derive.py

from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd
import re

from property_adviser.core.app_logging import log, warn
from property_adviser.core.runner import run_step


# --- Reusable helpers -------------------------------------------------
def _has_cols(df: pd.DataFrame, cols: Sequence[str]) -> tuple[bool, list[str]]:
    missing = [c for c in cols if c not in df.columns]
    return (len(missing) == 0, missing)

def _safe_ratio(n: pd.Series, d: pd.Series) -> pd.Series:
    out = np.where(d == 0, np.nan, n / d)
    return pd.Series(out, index=n.index).replace([np.inf, -np.inf], np.nan)

def _norm_token(s: str) -> str:
    """Uppercase, strip punctuation, collapse spaces."""
    if not isinstance(s, str):
        return ""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9& ]+", " ", s)   # keep & for 'BIGGIN & SCOTT'
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --- Feature Extraction ---
def extract_street(address: str, cfg: Dict[str, Any]) -> str:
    """Extract and clean street name from a full address string."""
    unknown = cfg.get("unknown_value", "Unknown")
    if not isinstance(address, str):
        return unknown

    addr = address.strip()

    # Step 1: If commas exist, take the part after the last comma
    if "," in addr:
        addr = addr.split(",")[-1].strip()

    # Step 2: Remove "unit-like" tokens followed by slash-separated numbers (e.g. "Flat 12/34")
    unit_slash_pattern = r"^(unit|flat|apt|apartment|lot|ste|suite|shop|villa|studio|duplex)\s*\d+[a-zA-Z]?/\d+[a-zA-Z]?\s*"
    addr = re.sub(unit_slash_pattern, "", addr, flags=re.IGNORECASE)

    # Step 3: Remove other common unit/house prefixes (with optional punctuation)
    unit_patterns = [
        r"^(unit|flat|apt|apartment|lot|ste|suite|shop|villa|studio|duplex)\s*[0-9a-z]+\s*-?\s*",
        r"^(unit|flat|apt|apartment|lot|ste|suite|shop|villa|studio|duplex)\s*[0-9a-z]+\s*,?\s*",
    ]
    for pattern in unit_patterns:
        addr = re.sub(pattern, "", addr, flags=re.IGNORECASE)

    # Step 4: Strip leading numeric/compound numbers
    numeric_patterns = [
        r"^\d+[a-zA-Z]?-\d+[a-zA-Z]?\s+",
        r"^\d+[a-zA-Z]?/\d+[a-zA-Z]?\s+",
        r"^\d+[a-zA-Z]?-\d+[a-zA-Z]?/\d+\s+",
        r"^\d+[a-zA-Z]?\s+",
        r"^\d+\s+",
    ]
    for _ in range(3):
        original = addr
        for pattern in numeric_patterns:
            addr = re.sub(pattern, "", addr)
        if addr == original:
            break

    # Step 5: Remove leading separators
    addr = re.sub(r"^-\s+", "", addr)

    # Step 6: Final cleanup
    addr = addr.strip().rstrip(",")

    return addr.title() if addr else unknown


# --- Derived Features ---
def apply_group_aggregate(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Aggregate a column by groups and merge the result back into the dataframe.

    Required in spec:
      - group_by: list[str]
      - target: str
      - aggregate: str | callable (e.g., "mean", "median", "sum")
      - output: str (name of new column)
    """
    if not spec.get("enabled", True):
        return df

    missing_keys = [k for k in ("group_by", "target", "aggregate", "output") if k not in spec]
    if missing_keys:
        warn("derive.groupby_aggregate", reason="missing_keys", keys=missing_keys)
        return df

    group_by: List[str] = spec["group_by"]
    target: str = spec["target"]
    agg = spec["aggregate"]
    output: str = spec["output"]

    if any(col not in df.columns for col in group_by + [target]):
        missing_cols = [c for c in group_by + [target] if c not in df.columns]
        warn("derive.groupby_aggregate", reason="missing_columns", missing=missing_cols)
        return df

    if output in df.columns:
        warn("derive.groupby_aggregate", reason="output_exists", output=output)
        return df

    aggregated = (
        df[group_by + [target]].groupby(group_by, dropna=False)[target].agg(agg).reset_index(name=output)
    )
    out_df = df.merge(aggregated, on=group_by, how="left")
    log("derive.groupby_aggregate", output=output, rows=out_df.shape[0], groups=len(aggregated))
    return out_df


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

    log(
        "derive.price_factor",
        numerator=num,
        denominator=den_col,
        output=out,
        min_value=min_val,
        rows_in=before,
        rows_out=df.shape[0],
    )
    return df


def _derive_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year/month components from 'saleDate' (if present)."""
    if "saleDate" not in df.columns:
        return df

    sd = pd.to_datetime(df["saleDate"], errors="coerce")
    if sd.notna().any():
        df["saleYear"] = sd.dt.year
        df["saleMonth"] = sd.dt.month
        df["saleYearMonth"] = sd.dt.year * 100 + sd.dt.month
        log("derive.date_parts", cols=["saleYear", "saleMonth", "saleYearMonth"])
    return df


def _derive_property_category(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Derive a canonical propertyCategory field from propertyType + landUse.

    Categories (Australian usage):

    - House:
        Free-standing dwellings on their own lot. Detached houses with land.
        Includes 'HOUSE', 'HOUSE: STANDARD', 'HOUSE: ONE STOREY / LOWSET'.
        Land use: 'Detached Dwelling', 'Detached Dwelling (existing)'.

    - Unit:
        Medium-density dwellings such as townhouses, villas, duplexes.
        Includes 'UNIT', 'UNIT: STANDARD', 'UNIT: TOWNHOUSE/VILLA'.
        Land use: 'Townhouse', 'Single Strata Unit/Villa Unit/Townhouse',
        'OYO Unit', 'OYO Subdivided Dwelling'.

    - Flat:
        Older/smaller apartment blocks or walk-ups (e.g. 'FLATS: SELF CONTAINED').

    - Apartment:
        Multi-storey buildings, shared entries (often captured by land use).
        Land use: 'Strata Unit or Flat  (Unspecified)', 'OYO Subdivided Flat'.

    - Other:
        Commercial/industrial/mixed/vacant, or anything not matched above.
    """
    prop_col = spec.get("property_type_col", "propertyType")
    land_col = spec.get("land_use_col", "landUse")
    out_col = spec.get("output", "propertyCategory")
    unknown = spec.get("unknown_value", "Other")

    def classify(prop: str | None, land: str | None) -> str:
        p = (prop or "").strip().upper()
        l = (land or "").strip().upper()

        # --- House ---
        if p in {"HOUSE", "HOUSE: STANDARD", "HOUSE: ONE STOREY / LOWSET"}:
            return "House"
        if l in {"DETACHED DWELLING", "DETACHED DWELLING (EXISTING)"}:
            return "House"

        # --- Unit ---
        if p in {"UNIT", "UNIT: STANDARD", "UNIT: TOWNHOUSE/VILLA"}:
            return "Unit"
        if l in {
            "TOWNHOUSE",
            "SINGLE STRATA UNIT/VILLA UNIT/TOWNHOUSE",
            "OYO UNIT",
            "OYO SUBDIVIDED DWELLING",
        }:
            return "Unit"

        # --- Flat (check before Apartment) ---
        if "FLAT" in p:  # catches 'FLATS: SELF CONTAINED' etc.
            return "Flat"

        # --- Apartment ---
        if "APARTMENT" in p or "HIGH-RISE" in p or "MID-RISE" in p:
            return "Apartment"
        if l in {"STRATA UNIT OR FLAT  (UNSPECIFIED)", "OYO SUBDIVIDED FLAT"}:
            return "Apartment"

        # --- Other ---
        if p.startswith("BUSINESS") or p.startswith("COMMERCIAL") or p == "OTHER":
            return "Other"
        if l in {
            "UNKNOWN",
            "OTHER",
            "VACANT RESIDENTIAL DWELLING SITE/SURVEYED LOT",
            "RES LAND (WITH BUILDINGS THAT ADD NO VALUE)",
            "GENERAL PURPOSE WAREHOUSE",
            "GENERAL PURPOSE FACTORY",
            "RETAIL PREMISES (SINGLE OCCUPANCY)",
            "OFFICE PREMISES",
        }:
            return "Other"

        return unknown

    df[out_col] = df.apply(lambda r: classify(r.get(prop_col), r.get(land_col)), axis=1)
    log("derive.property_category", output=out_col, rows=df.shape[0])
    return df

def derive_month_cyclical(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Create cyclic encodings for month to capture seasonality.
    Spec:
      month_col: str (default 'saleMonth')
      out_prefix: str (default 'saleMonth')
    Outputs: <prefix>_sin, <prefix>_cos
    """
    if not spec.get("enabled", True):
        return df

    month_col = spec.get("month_col", "saleMonth")
    prefix = spec.get("out_prefix", month_col)

    ok, missing = _has_cols(df, [month_col])
    if not ok:
        warn("derive.month_cyclical", reason="missing_columns", missing=missing)
        return df

    # ensure ints 1..12
    m = pd.to_numeric(df[month_col], errors="coerce")
    rad = 2 * np.pi * (m.astype(float) % 12) / 12.0
    df[f"{prefix}_sin"] = np.sin(rad)
    df[f"{prefix}_cos"] = np.cos(rad)
    log("derive.month_cyclical", cols=[f"{prefix}_sin", f"{prefix}_cos"], rows=df.shape[0])
    return df

def derive_property_age(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute property age at time of sale.
    Spec:
      year_built_col: str (default 'yearBuilt')
      sale_year_col: str (default 'saleYear')  # created by _derive_date_parts
      output: str (default 'propertyAge')
      min_age: float (optional, e.g. 0), max_age: float (optional)
    """
    if not spec.get("enabled", True):
        return df

    yb = spec.get("year_built_col", "yearBuilt")
    sy = spec.get("sale_year_col", "saleYear")
    out = spec.get("output", "propertyAge")
    ok, missing = _has_cols(df, [yb, sy])
    if not ok:
        warn("derive.property_age", reason="missing_columns", missing=missing)
        return df

    age = pd.to_numeric(df[sy], errors="coerce") - pd.to_numeric(df[yb], errors="coerce")
    if "min_age" in spec:
        age = age.where(age >= float(spec["min_age"]))
    if "max_age" in spec:
        age = age.where(age <= float(spec["max_age"]))
    df[out] = age
    log("derive.property_age", output=out, rows=df.shape[0])
    return df

def derive_ratio(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute a ratio new_col = numerator / denominator.
    Spec:
      numerator: str
      denominator: str
      output: str
      clip_min, clip_max: float (optional)
      drop_na: bool (default False) -> drop rows where result is NA or 0/inf after ratio
    """
    if not spec.get("enabled", True):
        return df

    num, den, out = spec["numerator"], spec["denominator"], spec["output"]
    clip_min = spec.get("clip_min")
    clip_max = spec.get("clip_max")
    drop_na = bool(spec.get("drop_na", False))

    ok, missing = _has_cols(df, [num, den])
    if not ok:
        warn("derive.ratio", reason="missing_columns", missing=missing)
        return df

    result = _safe_ratio(pd.to_numeric(df[num], errors="coerce"),
                         pd.to_numeric(df[den], errors="coerce"))
    if clip_min is not None:
        result = result.where(result >= float(clip_min))
    if clip_max is not None:
        result = result.where(result <= float(clip_max))

    df[out] = result
    rows_in = df.shape[0]
    if drop_na:
        df = df[df[out].notna()].copy()
    log("derive.ratio", numerator=num, denominator=den, output=out, rows_in=rows_in, rows_out=df.shape[0])
    return df

def derive_price_per_area(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Wrapper for price per area.
    Spec:
      price_col: str (default 'salePrice')
      area_col: str (required)
      output: str (e.g., 'pricePerSqmLand')
      clip_min, clip_max, drop_na: see derive_ratio
    """
    if not spec.get("enabled", True):
        return df
    s = spec.copy()
    s.setdefault("price_col", "salePrice")
    if "area_col" not in s or "output" not in s:
        warn("derive.price_per_area", reason="missing_keys", keys=["area_col","output"])
        return df
    s["numerator"] = s["price_col"]
    s["denominator"] = s["area_col"]
    s["output"] = s["output"]
    return derive_ratio(df, s)


# --- Main Entry ---
def derive_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Run configured derivation steps on the dataframe."""
    df = _derive_date_parts(df)

    for name, spec in config.items():
        if not spec.get("enabled", True):
            log("derive.step", status="skipped", name=name, reason="disabled")
            continue

        if name == "street":
            src, out = spec["source"], spec["output"]
            drop_unknown = spec.get("drop_unknown", False)
            cfg = spec.get("config", {"unknown_value": "Unknown"})

            def _street(dfin: pd.DataFrame) -> pd.DataFrame:
                if src in dfin.columns:
                    before = dfin.shape[0]
                    dfin[out] = dfin[src].apply(lambda x: extract_street(x, cfg))
                    if drop_unknown:
                        dfin = dfin[dfin[out] != cfg["unknown_value"]].copy()
                    log(
                        "derive.street",
                        source=src,
                        output=out,
                        drop_unknown=drop_unknown,
                        rows_in=before,
                        rows_out=dfin.shape[0],
                    )
                else:
                    warn("derive.street", reason="missing_source", source=src)
                return dfin

            df = run_step("derive.street", _street, df)
            continue

        # Generic groupby aggregate (e.g., year_month_mean_price)
        if all(k in spec for k in ("group_by", "target", "aggregate", "output")):
            df = run_step("derive.group_aggregate", apply_group_aggregate, df, spec=spec)
            continue

        if name == "postcode_prefix":
            src, out, length = spec["source"], spec["output"], int(spec["length"])

            def _postcode(dfin: pd.DataFrame) -> pd.DataFrame:
                if src not in dfin.columns or out in dfin.columns:
                    return dfin
                dfin[out] = dfin[src].apply(
                    lambda v: str(v).strip()[:length] if pd.notna(v) else None
                )
                return dfin

            df = run_step("derive.postcode_prefix", _postcode, df)
            continue

        if name == "price_factor":
            df = run_step("derive.price_factor", derive_price_factor, df, spec=spec)
            continue

        if name == "property_category":
            df = run_step("derive.property_category", _derive_property_category, df, spec=spec)
            continue

        if name == "month_cyclical":
            df = run_step("derive.month_cyclical", derive_month_cyclical, df, spec=spec)
            continue

        if name == "property_age":
            df = run_step("derive.property_age", derive_property_age, df, spec=spec)
            continue

        if name == "ratio":
            df = run_step("derive.ratio", derive_ratio, df, spec=spec)
            continue

        if name == "price_per_area":
            df = run_step("derive.price_per_area", derive_price_per_area, df, spec=spec)
            continue

        warn("derive.step", name=name, reason="unknown_step")

    return df
