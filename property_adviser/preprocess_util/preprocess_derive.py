# preprocess_derive.py

from typing import Any, Dict
import numpy as np
import pandas as pd
import re

from src.common.app_logging import log, warn
from src.common.runner import run_step


# --- Feature Extraction ---
def extract_street(address: str, cfg: Dict[str, Any]) -> str:
    """Extract and clean street name from a full address string."""
    unknown = cfg.get("unknown_value", "Unknown")
    if not isinstance(address, str):
        return unknown
    
    # Work with a copy of the address
    addr = address.strip()
    
    # Step 1: Handle addresses with commas (take the part after the last comma)
    if ',' in addr:
        addr_parts = addr.split(',')
        addr = addr_parts[-1].strip()
    
    # Step 2: Special handling for unit types followed by slash-separated numbers
    # This specifically targets formats like "Flat 12/34", "Suite 101/500", etc.
    unit_slash_pattern = r"^(unit|flat|apt|apartment|lot|ste|suite|shop|villa|studio|duplex)\s*\d+[a-zA-Z]?[/]\d+[a-zA-Z]?\s*"
    addr = re.sub(unit_slash_pattern, "", addr, flags=re.IGNORECASE)
    
    # Step 3: Remove other common unit/house prefixes with optional punctuation
    unit_patterns = [
        r"^(unit|flat|apt|apartment|lot|ste|suite|shop|villa|studio|duplex)\s*[0-9a-z]+\s*[-]?\s*",
        r"^(unit|flat|apt|apartment|lot|ste|suite|shop|villa|studio|duplex)\s*[0-9a-z]+\s*[,]?\s*",
    ]
    for pattern in unit_patterns:
        addr = re.sub(pattern, "", addr, flags=re.IGNORECASE)
    
    # Step 4: Remove numeric parts at the beginning, including complex formats
    # Handle various number formats: simple, range, unit/house
    numeric_patterns = [
        # Number range with hyphen (724-728 or 12-14A)
        r"^\d+[a-zA-Z]?[-]\d+[a-zA-Z]?\s+",
        # Unit/house number format with slash (45A/67 or 101/500)
        r"^\d+[a-zA-Z]?[/]\d+[a-zA-Z]?\s+",
        # Complex format with both hyphen and slash (12-14A/100)
        r"^\d+[a-zA-Z]?[-]\d+[a-zA-Z]?[/]\d+\s+",
        # Simple numeric prefixes (123 or 123a)
        r"^\d+[a-zA-Z]?\s+",
        # Any remaining numeric parts
        r"^\d+\s+",
    ]
    
    # Apply numeric patterns multiple times to handle nested cases
    # We'll do up to 3 passes to avoid infinite loops
    for _ in range(3):
        original = addr
        for pattern in numeric_patterns:
            addr = re.sub(pattern, "", addr)
        # If no changes after one full pass, we're done
        if addr == original:
            break
    
    # Step 5: Remove any remaining hyphens or other separators at the beginning
    addr = re.sub(r"^[-]\s+", "", addr)
    
    # Step 6: Final cleanup
    addr = addr.strip().rstrip(',')
    
    return addr.title() if addr else unknown

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