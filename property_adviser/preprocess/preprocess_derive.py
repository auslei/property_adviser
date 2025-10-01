from typing import Any, Dict, List, Sequence, Optional, Tuple
import numpy as np
import pandas as pd
import re

from property_adviser.core.app_logging import log, warn
from property_adviser.core.runner import run_step
from property_adviser.macro.macro_data import add_macro_yearly


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


def _slugify_tag(value: Optional[str], fallback: str = "other") -> str:
    """Convert a property-type label to a safe slug.

    Parameters
    ----------
    value:
        Raw property-type label (may be None).
    fallback:
        Value to use when the slug would be empty.
    """
    if not value:
        return fallback
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug or fallback


def _apply_bucket_definitions(df: pd.DataFrame, buckets_cfg: Dict[str, Any]) -> pd.DataFrame:
    if not buckets_cfg:
        return df

    df = df.copy()

    for bucket_name, spec in buckets_cfg.items():
        if not isinstance(spec, dict) or not spec.get("enabled", True):
            continue

        source = spec.get("source")
        if not source or source not in df.columns:
            warn("derive.bucket", bucket=bucket_name, reason="missing_source", source=source)
            continue

        output = spec.get("output", f"{bucket_name}_bucket")
        fill_value = spec.get("fill_value", "Unknown")

        series = df[source]
        bucket_values: pd.Series

        if "bins" in spec:
            raw_values = pd.to_numeric(series, errors="coerce")
            bins = spec.get("bins") or []
            if not isinstance(bins, (list, tuple)) or not bins:
                warn("derive.bucket", bucket=bucket_name, reason="invalid_bins")
                continue
            bins = sorted(float(b) for b in bins)
            include_lowest = bool(spec.get("include_lowest", True))
            right = bool(spec.get("right", True))
            extend_lower = spec.get("extend_lower", True)
            extend_upper = spec.get("extend_upper", True)

            edges: List[float] = bins.copy()
            if extend_lower:
                edges = [-np.inf] + edges
            if extend_upper:
                edges = edges + [np.inf]
            labels = spec.get("labels")
            if labels is not None and len(labels) != len(edges) - 1:
                warn(
                    "derive.bucket",
                    bucket=bucket_name,
                    reason="label_mismatch",
                    expected=len(edges) - 1,
                    provided=len(labels),
                )
                labels = None
            bucket_values = pd.cut(
                raw_values,
                bins=edges,
                labels=labels,
                include_lowest=include_lowest,
                right=right,
            )
            if labels is None:
                bucket_values = bucket_values.astype(str)

        elif "mapping" in spec:
            mapping = spec.get("mapping") or {}
            default = mapping.get("default", fill_value)
            bucket_values = series.map(mapping).fillna(default)
        else:
            warn("derive.bucket", bucket=bucket_name, reason="unsupported_spec")
            continue

        bucket_series = bucket_values.astype(str).replace({"nan": np.nan}).fillna(fill_value)
        df[output] = bucket_series
        log("derive.bucket", bucket=bucket_name, output=output, unique=int(df[output].nunique()))

    return df


def _join_macro_features(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    if not spec or not spec.get("enabled", True):
        return df

    path = spec.get("path") or spec.get("macro_path") or "data/macro/macro_au_annual.csv"
    sale_year_col = spec.get("sale_year_col", "saleYear")

    try:
        joined = add_macro_yearly(df, macro_path=path, sale_year_col=sale_year_col)
    except FileNotFoundError:
        warn("derive.macro_join", reason="missing_file", path=path)
        return df
    except KeyError as exc:
        warn("derive.macro_join", reason="missing_sale_year", error=str(exc))
        return df

    log(
        "derive.macro_join",
        path=path,
        sale_year_col=sale_year_col,
        rows=joined.shape[0],
        cols_added=len(set(joined.columns) - set(df.columns)),
    )
    return joined


def _compute_future_targets(
    df: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    month_col: str,
    specs: Sequence[Dict[str, Any]],
) -> Optional[pd.DataFrame]:
    if not specs:
        return None

    month_numeric = pd.to_numeric(df[month_col], errors="coerce")
    working = df.copy()
    working[month_col] = month_numeric

    results: List[pd.Series] = []
    for spec in specs:
        if not spec.get("name"):
            continue
        source = spec.get("source")
        if source not in working.columns:
            warn("derive.future_target", target=spec.get("name"), reason="missing_source", source=source)
            continue
        agg = spec.get("agg", "median")
        horizon = int(spec.get("horizon", 6))
        window = int(spec.get("window", 1))
        min_periods = int(spec.get("min_periods", window))

        grouped = working.groupby(list(group_keys) + [month_col], dropna=False)[source].agg(agg)
        grouped.index.names = list(group_keys) + [month_col]
        grouped = grouped.sort_index()

        def compute(series: pd.Series) -> pd.Series:
            series = series.sort_index()
            shifted = series.shift(-horizon)
            if window > 1:
                shifted = shifted.rolling(window=window, min_periods=min_periods).mean()
            return shifted

        future_series = grouped.groupby(level=list(range(len(group_keys))), group_keys=False).apply(compute)
        future_series.name = spec["name"]
        results.append(future_series)

    if not results:
        return None

    combined = pd.concat(results, axis=1)
    combined = combined.reset_index()
    return combined


def build_segments(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    grouping_cfg = config.get("grouping") or {}
    if not grouping_cfg or not grouping_cfg.get("enabled", True):
        return None, {}

    month_col = grouping_cfg.get("month_col", "saleYearMonth")
    keys: List[str] = grouping_cfg.get("keys", [])
    min_rows = int(grouping_cfg.get("min_rows", 0))
    min_months = int(grouping_cfg.get("min_months", 0))

    missing = [col for col in keys + [month_col] if col not in df.columns]
    if missing:
        warn("derive.segment", reason="missing_columns", missing=missing)
        return None, {}

    working = df.copy()
    working[month_col] = pd.to_numeric(working[month_col], errors="coerce")

    group_cols = keys + [month_col]

    segment_df = (
        working.groupby(group_cols, dropna=False)
        .size()
        .rename("record_count")
        .reset_index()
    )

    agg_cfg = config.get("aggregations") or {}
    metrics = agg_cfg.get("metrics", []) if agg_cfg.get("enabled", True) else []
    for metric in metrics:
        name = metric.get("name")
        source = metric.get("source")
        agg = metric.get("agg", "median")
        if not name or not source:
            continue
        if source not in working.columns:
            warn("derive.segment", metric=name, reason="missing_source", source=source)
            continue
        agg_series = (
            working.groupby(group_cols, dropna=False)[source]
            .agg(agg)
            .rename(name)
            .reset_index()
        )
        segment_df = segment_df.merge(agg_series, on=group_cols, how="left")

    carry_cfg = agg_cfg.get("carry") or {}
    carry_enabled = carry_cfg.get("enabled")
    if carry_enabled is None:
        carry_enabled = bool(metrics)

    if carry_enabled:
        explicit_cols = [col for col in carry_cfg.get("columns", []) if isinstance(col, str)]
        prefix_list = [pref for pref in carry_cfg.get("prefixes", []) if isinstance(pref, str)]
        auto_detect = bool(carry_cfg.get("auto_detect", False))
        max_unique = int(carry_cfg.get("max_unique", 1))
        allow_mixed = bool(carry_cfg.get("allow_mixed", False))

        candidate_cols = set(explicit_cols)
        if prefix_list:
            for col in working.columns:
                if any(col.startswith(pref) for pref in prefix_list):
                    candidate_cols.add(col)

        if auto_detect:
            excluded = set(group_cols) | set(segment_df.columns)
            for col in working.columns:
                if col in excluded or col in candidate_cols:
                    continue
                candidate_cols.add(col)

        grouped = working.groupby(group_cols, dropna=False)
        safe_columns: list[str] = []
        rejected_columns: list[str] = []

        for col in sorted(candidate_cols):
            if col in group_cols or col in segment_df.columns:
                continue
            if col not in working.columns:
                warn(
                    "derive.segment",
                    reason="carry_missing_column",
                    column=col,
                )
                continue

            unique_counts = grouped[col].nunique(dropna=False)
            max_count = unique_counts.max()
            if pd.isna(max_count):
                # Column was entirely NA; include to mirror original behaviour
                safe_columns.append(col)
                continue

            if max_count <= max_unique:
                safe_columns.append(col)
            else:
                rejected_columns.append(col)
                if allow_mixed:
                    safe_columns.append(col)

        if rejected_columns:
            warn(
                "derive.segment",
                reason="carry_conflict",
                columns=rejected_columns,
                max_unique=max_unique,
                allow_mixed=allow_mixed,
            )

        if safe_columns:
            carry_frame = grouped[safe_columns].first().reset_index()
            segment_df = segment_df.merge(carry_frame, on=group_cols, how="left")
            log(
                "derive.segment_carry",
                columns=len(safe_columns),
                explicit=len(explicit_cols),
                prefixes=len(prefix_list),
                auto_detect=auto_detect,
            )

    # Filter groups by overall activity (rows/months)
    if min_rows or min_months:
        group_totals = working.groupby(keys, dropna=False).size()
        group_months = working.groupby(keys, dropna=False)[month_col].nunique()
        valid_groups = group_totals.index
        if min_rows:
            valid_groups = group_totals[group_totals >= min_rows].index
        if min_months:
            valid_groups = group_months[group_months >= min_months].index.intersection(valid_groups)
        if len(valid_groups) != len(group_totals):
            segment_df = segment_df[segment_df.set_index(keys).index.isin(valid_groups)]

    future_specs = config.get("future_targets") or []
    future_df = _compute_future_targets(df, group_keys=keys, month_col=month_col, specs=future_specs)
    if future_df is not None:
        segment_df = segment_df.merge(future_df, on=group_cols, how="left")

    # Optional dropping of rows with missing future targets when drop_na flag set
    for spec in future_specs:
        target_name = spec.get("name")
        if spec.get("drop_na") and target_name in segment_df.columns:
            segment_df = segment_df[segment_df[target_name].notna()]
        smooth_flag = spec.get("smooth")
        if smooth_flag:
            smooth_col = f"{target_name}_smooth"
            if smooth_col in segment_df.columns:
                segment_df = segment_df[segment_df[smooth_col].notna()]


    # ------------------------------------------------------------------
    # Trend / relative features
    # ------------------------------------------------------------------
    if keys:
        segment_df = segment_df.sort_values(keys + [month_col])
        group = segment_df.groupby(keys, sort=False)

        # Rolling statistics for current price median
        current_col = "current_price_median"
        if current_col in segment_df.columns:
            rolling_mean_12 = group[current_col].transform(lambda s: s.rolling(window=12, min_periods=3).mean())
            rolling_std_12 = group[current_col].transform(lambda s: s.rolling(window=12, min_periods=3).std())
            lag_12 = group[current_col].shift(12)
            lag_6 = group[current_col].shift(6)

            segment_df["current_price_median_roll_mean_12m"] = rolling_mean_12
            segment_df["current_price_median_roll_std_12m"] = rolling_std_12
            segment_df["current_price_median_z_12m"] = (
                (segment_df[current_col] - rolling_mean_12)
                / rolling_std_12.replace({0.0: np.nan})
            )
            segment_df["current_price_median_yoy"] = (
                segment_df[current_col] / lag_12.replace({0.0: np.nan})
            ) - 1
            segment_df["current_price_median_6m_change"] = (
                segment_df[current_col] / lag_6.replace({0.0: np.nan})
            ) - 1

    if "current_price_median" in segment_df.columns and "suburb_price_median_current" in segment_df.columns:
        segment_df["current_price_median_rel_suburb"] = (
            segment_df["current_price_median"]
            / segment_df["suburb_price_median_current"].replace({0.0: np.nan})
        )

    base_current = segment_df.get("current_price_median")
    if base_current is not None:
        safe_base = base_current.replace({0.0: np.nan})
        grouped = segment_df.groupby(keys, sort=False) if keys else None
        for spec in future_specs:
            name = spec.get("name")
            if not name or name not in segment_df.columns:
                continue
            future_vals = segment_df[name]
            segment_df[f"{name}_delta"] = future_vals / safe_base - 1
            segment_df[f"{name}_diff"] = future_vals - base_current

            if grouped is not None and spec.get("smooth"):
                window = int(spec.get("smooth", 12))
                smooth_col = f"{name}_smooth"
                smooth_vals = grouped[name].transform(
                    lambda s: s.rolling(window=window, min_periods=max(3, window // 2)).mean()
                )
                segment_df[smooth_col] = smooth_vals
                segment_df[f"{smooth_col}_delta"] = smooth_vals / safe_base - 1
                segment_df[f"{smooth_col}_diff"] = smooth_vals - base_current

    for spec in future_specs:
        delta_col = f"{spec.get('name')}_delta"
        if spec.get("drop_na") and delta_col in segment_df.columns:
            segment_df = segment_df[segment_df[delta_col].notna()]

        if spec.get("smooth"):
            smooth_delta = f"{spec.get('name')}_smooth_delta"
            if spec.get("drop_na") and smooth_delta in segment_df.columns:
                segment_df = segment_df[segment_df[smooth_delta].notna()]

    segment_df = segment_df.reset_index(drop=True)
    segment_df["observingYearMonth"] = segment_df[month_col]

    segment_meta = {
        "grouping_keys": keys,
        "month_column": month_col,
        "rows": int(segment_df.shape[0]),
        "groups": int(segment_df[keys].drop_duplicates().shape[0]) if keys else int(segment_df.shape[0]),
        "future_targets": {
            spec.get("name"): int(segment_df[spec.get("name")].notna().sum())
            for spec in future_specs
            if spec.get("name") in segment_df.columns
        },
    }

    return segment_df, segment_meta


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

    # Step 4b: Drop leading tokens that still contain digits (e.g. "10/6-10")
    parts = addr.split()
    while parts and any(char.isdigit() for char in parts[0]):
        parts.pop(0)
    addr = " ".join(parts)

    # Step 5: Remove leading separators
    addr = re.sub(r"^-\s+", "", addr)

    # Step 6: Final cleanup
    addr = addr.strip().rstrip(",")

    return addr.title() if addr else unknown


# --- Derived Features (existing) ---
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

        # --- Flat ---
        if "FLAT" in p:
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
    num_offset = float(spec.get("numerator_offset", 0.0) or 0.0)
    den_offset = float(spec.get("denominator_offset", 0.0) or 0.0)
    den_min = spec.get("denominator_min")

    ok, missing = _has_cols(df, [num, den])
    if not ok:
        warn("derive.ratio", reason="missing_columns", missing=missing)
        return df

    num_series = pd.to_numeric(df[num], errors="coerce")
    den_series = pd.to_numeric(df[den], errors="coerce")

    if num_offset:
        num_series = num_series + num_offset
    if den_offset:
        den_series = den_series + den_offset
    if den_min is not None:
        den_series = den_series.where(den_series.abs() >= float(den_min))

    result = _safe_ratio(num_series, den_series)
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


# --- New: Temporal / Market Trend helpers -----------------------------
DERIVE_FN_DISPATCH: Dict[str, Any] = {
    "ratio": derive_ratio,
    "price_per_area": derive_price_per_area,
}


def derive_month_index(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a monotonic month index from saleYearMonth:
      month_id = dense rank of saleYearMonth starting at 0 (or offset).
    """
    if not spec.get("enabled", True):
        return df
    month_col = spec.get("month_col", "saleYearMonth")
    out = spec.get("output", "month_id")
    offset = int(spec.get("offset", 0))

    ok, missing = _has_cols(df, [month_col])
    if not ok:
        warn("derive.month_index", reason="missing_columns", missing=missing)
        return df

    # Dense rank, stable across dataset
    order = pd.Series(pd.unique(pd.Series(df[month_col]).dropna())).sort_values().tolist()
    mapping = {m: i + offset for i, m in enumerate(order)}
    df[out] = pd.to_numeric(df[month_col], errors="coerce").map(mapping)
    log("derive.month_index", output=out, offset=offset, unique_months=len(order))
    return df


def _build_suburb_month_table(
    df: pd.DataFrame,
    suburb_col: str,
    month_col: str,
    price_col: str
) -> pd.DataFrame:
    """Aggregate to suburb x month level (median, count, std)."""
    ok, missing = _has_cols(df, [suburb_col, month_col, price_col])
    if not ok:
        raise KeyError(f"Missing columns for suburb-month table: {missing}")
    tmp = df[[suburb_col, month_col, price_col]].copy()
    tmp[month_col] = pd.to_numeric(tmp[month_col], errors="coerce")
    g = (
        tmp.groupby([suburb_col, month_col], dropna=False)[price_col]
        .agg(median="median", count="count", std="std")
        .reset_index()
    )
    return g


def _build_suburb_rollup_features(
    sm: pd.DataFrame,
    *,
    suburb_col: str,
    month_col: str,
    windows: Sequence[int],
    base_prefix: str,
    overall: bool,
) -> pd.DataFrame:
    """Compute lagged suburb-level metrics for a given aggregate table.

    Parameters
    ----------
    sm:
        Suburb-month aggregate table with columns ['median', 'count', 'std'].
    base_prefix:
        Prefix for the output feature names (e.g. 'suburb' or 'suburb_house').
    overall:
        When True, align naming with the historical single-segment outputs.
    """
    if sm.empty:
        return pd.DataFrame(columns=[suburb_col, month_col])

    sm = sm.sort_values([suburb_col, month_col])
    sm["median_prior"] = sm.groupby(suburb_col)["median"].shift(1)
    sm["count_prior"] = sm.groupby(suburb_col)["count"].shift(1)
    sm["std_prior"] = sm.groupby(suburb_col)["std"].shift(1)

    for window in windows:
        window = int(window)
        sm[f"median_roll_{window}m"] = sm.groupby(suburb_col)["median_prior"].transform(
            lambda s: s.rolling(window=window, min_periods=1).median()
        )
        sm[f"count_roll_{window}m"] = sm.groupby(suburb_col)["count_prior"].transform(
            lambda s: s.rolling(window=window, min_periods=1).sum()
        )
        sm[f"std_roll_{window}m"] = sm.groupby(suburb_col)["std_prior"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )

    sm["delta_3m"] = sm.groupby(suburb_col)["median_prior"].pct_change(periods=3)
    sm["delta_12m"] = sm.groupby(suburb_col)["median_prior"].pct_change(periods=12)

    # Replace infinities that can arise when the lagged median is zero.
    sm = sm.replace([np.inf, -np.inf], np.nan)

    keep_cols = [
        suburb_col,
        month_col,
        "median_prior",
        "count_prior",
        "std_prior",
        "median_roll_3m",
        "median_roll_6m",
        "median_roll_12m",
        "count_roll_3m",
        "count_roll_6m",
        "count_roll_12m",
        "std_roll_3m",
        "std_roll_6m",
        "std_roll_12m",
        "delta_3m",
        "delta_12m",
    ]
    sm = sm[keep_cols]

    rename = {
        "median_prior": f"{base_prefix}_price_median_current",
        "median_roll_3m": f"{base_prefix}_price_median_3m",
        "median_roll_6m": f"{base_prefix}_price_median_6m",
        "median_roll_12m": f"{base_prefix}_price_median_12m",
        "count_roll_3m": f"{base_prefix}_txn_count_3m",
        "count_roll_6m": f"{base_prefix}_txn_count_6m",
        "count_roll_12m": f"{base_prefix}_txn_count_12m",
        "std_roll_3m": f"{base_prefix}_volatility_3m",
        "std_roll_6m": f"{base_prefix}_volatility_6m",
        "std_roll_12m": f"{base_prefix}_volatility_12m",
        "delta_3m": f"{base_prefix}_delta_3m",
        "delta_12m": f"{base_prefix}_delta_12m",
    }

    if overall:
        rename["count_prior"] = "count"
        rename["std_prior"] = "std"
    else:
        rename["count_prior"] = f"{base_prefix}_txn_count_current"
        rename["std_prior"] = f"{base_prefix}_volatility_current"

    return sm.rename(columns=rename)


def derive_suburb_time_aggregates(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Rolling suburb medians, counts (volume), and volatility (std) with NO leakage:
      - shift(1) before applying rolling windows so only past months are used.
    Also adds price momentum deltas on monthly medians (pct_change over 3/12 months).

    Spec:
      suburb_col: str (default 'suburb')
      month_col: str (default 'saleYearMonth')
      price_col: str (default 'salePrice')
      windows: list[int] (default [3,6,12])
      prefix: str (default 'suburb')
    """
    if not spec.get("enabled", True):
        return df

    suburb_col = spec.get("suburb_col", "suburb")
    month_col = spec.get("month_col", "saleYearMonth")
    price_col = spec.get("price_col", "salePrice")
    windows: List[int] = spec.get("windows", [3, 6, 12])
    prefix = spec.get("prefix", "suburb")

    try:
        sm = _build_suburb_month_table(df, suburb_col, month_col, price_col)
    except KeyError as e:
        warn("derive.suburb_time_aggregates", reason="missing_columns", error=str(e))
        return df

    rollups_overall = _build_suburb_rollup_features(
        sm,
        suburb_col=suburb_col,
        month_col=month_col,
        windows=windows,
        base_prefix=prefix,
        overall=True,
    )

    merged = df.merge(rollups_overall, on=[suburb_col, month_col], how="left")

    tags_used: List[str] = []
    type_col = spec.get("type_col")
    if type_col:
        if type_col not in df.columns:
            warn("derive.suburb_time_aggregates", reason="missing_type_column", column=type_col)
        else:
            type_map_cfg = spec.get("type_map", {})
            type_map = {str(k).strip().upper(): _slugify_tag(v) for k, v in type_map_cfg.items()}
            type_default = _slugify_tag(spec.get("type_default", "other"))

            raw_types = df[type_col].fillna("").astype(str).str.strip().str.upper()
            mapped_types = raw_types.map(type_map)
            tag_series = mapped_types.fillna(type_default)

            configured_tags = spec.get("type_tags")
            if configured_tags:
                allowed_tags = []
                for tag in configured_tags:
                    slug = _slugify_tag(tag)
                    if slug and slug not in allowed_tags:
                        allowed_tags.append(slug)
            else:
                allowed_tags = sorted(tag for tag in tag_series.dropna().unique() if tag)

            type_subset = df[[suburb_col, month_col, price_col]].copy()
            type_subset["__type_tag"] = tag_series

            for tag in allowed_tags:
                if not tag:
                    continue
                subset = type_subset[type_subset["__type_tag"] == tag]
                if subset.empty:
                    continue
                sm_tag = _build_suburb_month_table(
                    subset[[suburb_col, month_col, price_col]],
                    suburb_col,
                    month_col,
                    price_col,
                )
                if sm_tag.empty:
                    continue
                tag_prefix = f"{prefix}_{tag}"
                rollups_tag = _build_suburb_rollup_features(
                    sm_tag,
                    suburb_col=suburb_col,
                    month_col=month_col,
                    windows=windows,
                    base_prefix=tag_prefix,
                    overall=False,
                )
                merged = merged.merge(rollups_tag, on=[suburb_col, month_col], how="left")
                tags_used.append(tag)

    log(
        "derive.suburb_time_aggregates",
        rows=merged.shape[0],
        windows=list(windows),
        type_tags=tags_used or None,
    )
    return merged


def derive_relative_pricing(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Relative pricing features:
      - price_vs_suburb_median = salePrice / suburb_price_median_current
      - price_vs_region_median (optional if region_col provided)
      - street_effect: street_mean / suburb_mean, only if street_count >= min_samples

    Spec:
      price_col: str (default 'salePrice')
      suburb_col: str (default 'suburb')
      month_col: str (default 'saleYearMonth')
      suburb_median_col: str (default 'suburb_price_median_current')
      region_col: Optional[str] (e.g., 'region')
      street_col: Optional[str] (e.g., 'street')
      min_samples: int (default 5) for street_effect
      output_prefix: str (default 'rel')
    """
    if not spec.get("enabled", True):
        return df

    price_col = spec.get("price_col", "salePrice")
    suburb_col = spec.get("suburb_col", "suburb")
    month_col = spec.get("month_col", "saleYearMonth")
    median_col = spec.get("suburb_median_col", "suburb_price_median_current")
    region_col = spec.get("region_col")
    street_col = spec.get("street_col")
    min_samples = int(spec.get("min_samples", 5))
    prefix = spec.get("output_prefix", "rel")

    needed = [price_col, suburb_col, month_col, median_col]
    ok, missing = _has_cols(df, needed)
    if not ok:
        warn("derive.relative_pricing", reason="missing_columns", missing=missing)
        return df

    # price vs suburb median
    df[f"{prefix}_price_vs_suburb_median"] = _safe_ratio(
        pd.to_numeric(df[price_col], errors="coerce"),
        pd.to_numeric(df[median_col], errors="coerce"),
    )

    # region median (per month) if region_col provided
    if region_col and region_col in df.columns:
        tmp = df[[region_col, month_col, price_col]].copy()
        tmp[month_col] = pd.to_numeric(tmp[month_col], errors="coerce")
        rmed = (
            tmp.groupby([region_col, month_col], dropna=False)[price_col]
            .median()
            .reset_index(name="region_median")
        )
        df = df.merge(rmed, on=[region_col, month_col], how="left")
        df[f"{prefix}_price_vs_region_median"] = _safe_ratio(
            pd.to_numeric(df[price_col], errors="coerce"),
            pd.to_numeric(df["region_median"], errors="coerce"),
        )
        df = df.drop(columns=["region_median"], errors="ignore")

    # street effect (global means with leave-one-out adjustment where possible)
    if street_col and street_col in df.columns:
        # Precompute sums and counts to allow LOO mean
        grp_cols = [street_col]
        s_sum = df.groupby(grp_cols, dropna=False)[price_col].transform("sum")
        s_cnt = df.groupby(grp_cols, dropna=False)[price_col].transform("count")
        st_mean_loo = (s_sum - df[price_col]) / (s_cnt - 1)
        st_mean_loo = st_mean_loo.where(s_cnt > 1)  # NaN if only one obs

        # Suburb mean (LOO)
        sub_sum = df.groupby([suburb_col], dropna=False)[price_col].transform("sum")
        sub_cnt = df.groupby([suburb_col], dropna=False)[price_col].transform("count")
        sub_mean_loo = (sub_sum - df[price_col]) / (sub_cnt - 1)
        sub_mean_loo = sub_mean_loo.where(sub_cnt > 1)

        effect = _safe_ratio(st_mean_loo, sub_mean_loo)
        # Enforce minimum sample requirement on street
        effect = effect.where(s_cnt >= min_samples)
        df[f"{prefix}_street_effect"] = effect

    log("derive.relative_pricing",
        created=[c for c in df.columns if c.startswith(prefix + "_")],
        min_samples=min_samples)
    return df


def derive_age_bands(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Bin property age into categories.
    Spec:
      age_col: str (default 'propertyAge')
      output: str (default 'propertyAgeBand')
      bands: list of upper bounds (default [5, 20]) giving: <=5, 6-20, >20
      labels: optional list[str]
    """
    if not spec.get("enabled", True):
        return df

    age_col = spec.get("age_col", "propertyAge")
    out = spec.get("output", "propertyAgeBand")
    bands = spec.get("bands", [5, 20])
    labels = spec.get("labels")

    ok, missing = _has_cols(df, [age_col])
    if not ok:
        warn("derive.age_bands", reason="missing_columns", missing=missing)
        return df

    age = pd.to_numeric(df[age_col], errors="coerce")
    edges = [-np.inf] + list(bands) + [np.inf]
    if labels is None:
        labels = [f"0–{bands[0]}", f"{bands[0]+1}–{bands[1]}", f"{bands[1]+1}+"]
    df[out] = pd.cut(age, bins=edges, labels=labels)
    log("derive.age_bands", output=out, bands=bands, labels=list(labels))
    return df


# --- Main Entry ---
def derive_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Run configured derivation steps on the dataframe."""
    df = _derive_date_parts(df)

    special_keys = {"buckets", "grouping", "aggregations", "future_targets"}

    for name, spec in config.items():
        if name in special_keys:
            continue
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

        # --- New entries ---
        if name == "month_index":
            df = run_step("derive.month_index", derive_month_index, df, spec=spec)
            continue

        if name == "suburb_time_aggregates":
            df = run_step("derive.suburb_time_aggregates", derive_suburb_time_aggregates, df, spec=spec)
            continue

        if name == "relative_pricing":
            df = run_step("derive.relative_pricing", derive_relative_pricing, df, spec=spec)
            continue

        if name == "age_bands":
            df = run_step("derive.age_bands", derive_age_bands, df, spec=spec)
            continue

        fn_name = str(spec.get("fn") or "").strip()
        if fn_name:
            fn_impl = DERIVE_FN_DISPATCH.get(fn_name)
            if fn_impl:
                df = run_step(f"derive.{fn_name}", fn_impl, df, spec=spec)
            else:
                warn("derive.step", name=name, reason="unknown_fn", fn=fn_name)
            continue

        warn("derive.step", name=name, reason="unknown_step")

    macro_cfg = config.get("macro_join")
    if macro_cfg:
        df = run_step("derive.macro_join", _join_macro_features, df, spec=macro_cfg)

    if "buckets" in config:
        df = run_step("derive.buckets", _apply_bucket_definitions, df, buckets_cfg=config.get("buckets", {}))


    return df
