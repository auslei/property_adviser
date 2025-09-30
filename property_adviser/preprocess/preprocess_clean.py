# src/preprocess_clean.py
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
import unicodedata

import numpy as np
import pandas as pd

from property_adviser.core.app_logging import log, warn, error

__all__ = ["clean_data"]  # only the pipeline is public

# ----------------------------
# Internal helpers (private)
# ----------------------------

def _normalize_text(value: str, replace_map: Dict[str, str]) -> str:
    """ASCII-normalise text, applying configured character replacements first."""
    for src, dst in replace_map.items():
        value = value.replace(src, dst).strip().upper().replace("  ", " ")
    value = unicodedata.normalize("NFKD", value)
    return value.encode("ascii", "ignore").decode("ascii")


def _clean_column_name(raw: str, used: Dict[str, int], replace_map: Dict[str, str]) -> str:
    """Turn raw column name into camelCase and ensure uniqueness within the frame."""
    name = _normalize_text(str(raw), replace_map).strip()
    tokens = [t for t in re.split(r"[^0-9a-zA-Z]+", name) if t] or ["column"]
    base = tokens[0].lower() + "".join(t.capitalize() for t in tokens[1:])
    n = used.get(base, 0)
    used[base] = n + 1
    return f"{base}{n + 1}" if n else base


def _filter_columns(df: pd.DataFrame, include: Optional[List[str]]) -> pd.DataFrame:
    """Keep only columns listed in `include` (ignore missing)."""
    if not include:
        return df
    keep = [c for c in include if c in df.columns]
    return df[keep]

def _has_cols(df: pd.DataFrame, cols: list[str]) -> tuple[bool, list[str]]:
    missing = [c for c in cols if c not in df.columns]
    return (len(missing) == 0, missing)

def _safe_ratio(n: pd.Series, d: pd.Series) -> pd.Series:
    out = n / d
    return out.replace([np.inf, -np.inf], np.nan)

def _norm_token(s: str) -> str:
    """Uppercase, strip punctuation, collapse spaces."""
    if not isinstance(s, str):
        return ""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9& ]+", " ", s)   # keep & for 'BIGGIN & SCOTT'
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- Derivation Functions -------------------------------------------------

def apply_category_mappings(
    df: pd.DataFrame,
    mappings: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply config-driven categorical consolidation.

    Supports two forms per key under `category_mappings`:

    A) Back-compat (your current style) — maps IN-PLACE:
       category_mappings:
         propertyType:
           House: [house, townhouse, dwelling]
           Unit:  [unit, villa, duplex]
           Apartment: [flat, studio]

    B) Extended (recommended for agency) — non-destructive:
       category_mappings:
         agencyBrand:
           source: agency
           default: Other
           mode: contains           # or "exact"
           rules:
             Noel Jones:     [NOEL JONES]
             Barry Plant:    [BARRY PLANT]
             Jellis Craig:   [JELLIS CRAIG]
             Ray White:      [RAY WHITE]
             Harcourts:      [HARCOURTS]
             Woodards:       [WOODARDS]
             Biggin & Scott: [BIGGIN & SCOTT]
             Fletchers:      [FLETCHERS]
             Unknown:        [UNKNOWN]
             Other:          [OTHER]
    """
    if not isinstance(mappings, dict) or not mappings:
        return df

    for out_or_col, spec in mappings.items():
        # -------- Extended form (with rules/source) --------
        if isinstance(spec, dict) and "rules" in spec:
            source = spec.get("source", out_or_col)
            default = spec.get("default", "Other")
            mode = spec.get("mode", "contains").lower()
            rules: Dict[str, List[str]] = spec["rules"]

            if source not in df.columns:
                warn("clean.category_map", reason="missing_source", source=source, output=out_or_col)
                continue

            base = df[source].astype(str).map(_norm_token)
            out = np.full(len(df), default, dtype=object)

            # Priority = order in YAML
            for canonical, keywords in rules.items():
                if not keywords:
                    continue
                kws = [ _norm_token(k) for k in keywords ]
                if mode == "exact":
                    mask = base.isin(kws)
                else:
                    # contains any keyword
                    mask = False
                    for kw in kws:
                        if kw:
                            mask = mask | base.str.contains(re.escape(kw), regex=True)
                out[mask] = canonical

            df[out_or_col] = out
            log("clean.category_map", output=out_or_col, source=source, unique=int(pd.Series(out).nunique()))
            continue

        # -------- Back-compat (map in-place on same column) --------
        if isinstance(spec, dict):
            col = out_or_col
            if col not in df.columns:
                warn("clean.category_map", reason="missing_column", column=col)
                continue
            base = df[col].astype(str).map(_norm_token)
            out = np.array(df[col].values, dtype=object)  # start with original values

            for canonical, keywords in spec.items():
                kws = [ _norm_token(k) for k in (keywords or []) ]
                mask = False
                for kw in kws:
                    if kw:
                        mask = mask | base.str.contains(re.escape(kw), regex=True)
                out[mask] = canonical

            df[col] = out
            log("clean.category_map_inplace", column=col, unique=int(pd.Series(out).nunique()))
            continue

        warn("clean.category_map", reason="bad_spec", key=out_or_col)

    return df


def _drop_mostly_empty_columns(df: pd.DataFrame, min_fraction: float) -> pd.DataFrame:
    """Drop columns with < `min_fraction` non-null values."""
    non_null_fraction = df.notna().mean()
    keep = non_null_fraction[non_null_fraction >= min_fraction].index
    return df.loc[:, keep]


def _to_numeric(s: pd.Series) -> pd.Series:
    """Strip common non-numeric characters and coerce to numeric (NaN on failure)."""
    cleaned = (
        s.astype(str)
        .str.replace(r"[,$]", "", regex=True)
        .str.replace(r"[^0-9.-]+", "", regex=True)
        .replace({"": np.nan, "-": np.nan, "nan": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_strings(s: pd.Series) -> pd.Series:
    """Trim and normalize common null-like tokens to NaN."""
    return s.astype(str).str.strip().replace(
        {"": np.nan, "-": np.nan, "nan": np.nan, "NaT": np.nan, "None": np.nan}
    )


def _bucket_categorical(s: pd.Series, min_count: int) -> pd.Series:
    """Bucket infrequent categories into 'Other' (preserve 'Unknown')."""
    counts = s.value_counts(dropna=False)
    allowed = set(counts[counts >= min_count].index)
    return s.apply(lambda v: v if v in allowed or v == "Unknown" else "Other")


def _load_data(data_path: str, pattern: str, encoding: str) -> pd.DataFrame:
    """Load and vertically concat all CSVs under a base path matching a pattern."""
    base = Path(data_path)
    csv_paths = sorted(base.glob(pattern))
    if not csv_paths:
        msg = f"No CSV files found in {base} matching pattern '{pattern}'."
        error("io.read_csv", error=msg, base=str(base), pattern=pattern)
        raise FileNotFoundError(msg)

    frames: List[pd.DataFrame] = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p, encoding=encoding)
            log("io.read_csv", file=str(p), rows=int(df.shape[0]), cols=int(df.shape[1]))
        except ValueError as ve:
            warn("io.read_csv", file=str(p), reason="usecols_mismatch", error=str(ve), fallback="read_all_cols")
            df = pd.read_csv(p, encoding=encoding)
            log("io.read_csv", file=str(p), rows=int(df.shape[0]), cols=int(df.shape[1]), mode="fallback_all_cols")

        df["__source_file"] = p.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log("io.concat", files=len(frames), rows=int(combined.shape[0]), cols=int(combined.shape[1]))
    return combined


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


# ---------- drop audit helpers ----------
_DROPPED_CHUNKS: List[pd.DataFrame] = []


def _filter_with_reason(df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
    """Keep rows where mask is True; record dropped rows with a reason."""
    if mask.dtype != bool or mask.shape[0] != df.shape[0]:
        raise ValueError("Mask must be boolean and aligned with dataframe")
    dropped = df.loc[~mask].copy()
    if not dropped.empty:
        dropped["_drop_reason"] = reason
        _DROPPED_CHUNKS.append(dropped)
    return df.loc[mask].copy()

def _emit_drop_audit(audit_path: Optional[Path]) -> None:
    """Write a parquet audit of dropped rows when a path is supplied."""
    if not _DROPPED_CHUNKS:
        return
    if audit_path is None:
        warn("rows.audit_skipped", reason="no_dropped_rows_path", dropped_chunks=len(_DROPPED_CHUNKS))
        return

    out = Path(audit_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit = pd.concat(_DROPPED_CHUNKS, ignore_index=True)
    audit.to_parquet(out, index=False)
    log("rows.audit_written", path=str(out), rows=int(audit.shape[0]))


# ----------------------------
# Public pipeline
# ----------------------------
def clean_data(config: Dict[str, Any], *, dropped_rows_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Phase 1: CLEANING
      1) Read CSVs (strictly from config.data_source)
      2) Rename columns
      3) Drop mostly-empty columns
      4) Apply category mappings
      5) Coerce numerics, normalise strings/dates
      6) Fill NaNs (non-target) BEFORE filtering required columns
      7) Filter rows missing required fields; filter target last
      8) Bucket high-cardinality categoricals (if configured)
    """
    # --- config (strict) ---
    _DROPPED_CHUNKS.clear()

    ds = config["data_source"]
    min_non_null = float(config["min_non_null_fraction"])
    special_to_ascii: Dict[str, str] = config["special_to_ascii"]

    # Required lists (no inline defaults)
    if "numeric_candidates" not in config:
        raise KeyError("Missing 'numeric_candidates' in cleaning config")

    combined = _load_data(ds["path"], ds["pattern"], ds["encoding"])

    # --- rename columns to camelCase (unique) ---
    used: Dict[str, int] = {}
    new_cols = [_clean_column_name(c, used, special_to_ascii) for c in combined.columns]
    combined.columns = new_cols
    log("columns.renamed", count=len(new_cols))

    # --- drop mostly-empty columns ---
    before_cols = combined.shape[1]
    combined = _drop_mostly_empty_columns(combined, min_non_null)
    log("columns.drop_mostly_empty", threshold=min_non_null, cols_in=before_cols, cols_out=int(combined.shape[1]))

    if config.get("category_mappings"):
        combined = apply_category_mappings(combined, config["category_mappings"])
        log("categorical.map", columns=list(config["category_mappings"].keys()))

    # --- numeric coercion (strictly from config-provided list) ---
    numeric_candidates: List[str] = list(config["numeric_candidates"])
    for col in numeric_candidates:
        if col in combined.columns:
            combined[col] = _to_numeric(combined[col])
    log("numeric.coerce", candidates=numeric_candidates)

    # --- replace non-ascii characters on selected columns (if configured) ---
    if config.get("normalise_text_columns"):
        for col in config["normalise_text_columns"]:
            if col in combined.columns:
                combined[col] = combined[col].apply(
                    lambda x: _normalize_text(str(x), special_to_ascii) if pd.notna(x) else x
                )
        log("normalise_text_columns", columns=list(config["normalise_text_columns"]))

    # --- date normalisation ---
    if "saleDate" in combined.columns:
        before = len(combined)
        combined["saleDate"] = pd.to_datetime(combined["saleDate"], errors="coerce")
        log("dates.sale_parsed", cols=["saleDate"], rows_in=before, rows_out=len(combined))

    # --- string normalisation across object/string columns ---
    for col in combined.select_dtypes(include=["object", "string"]).columns:
        combined[col] = _normalize_strings(combined[col])
    log("strings.normalised")

    # -----------------------------
    # FILL NaNs BEFORE required-filter
    # -----------------------------
    target = config.get("primary_target", "salePrice")
    numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        if c != target:  # do not impute the target here
            combined[c] = combined[c].fillna(combined[c].median())

    categorical_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    for c in categorical_cols:
        combined[c] = combined[c].fillna("Unknown")
    log("na.fill", numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    # --- filter rows based on required columns (AFTER fill) ---
    required_cols = config.get("required_columns", [])
    for col in required_cols:
        if col in combined.columns and col != target:
            before = int(combined.shape[0])
            combined = _filter_with_reason(combined, combined[col].notna(), f"{col}:required")
            log("rows.filter_required", column=col, rows_in=before, rows_out=int(combined.shape[0]))

    # --- finally filter target (cannot train without it) ---
    if target in combined.columns:
        before = int(combined.shape[0])
        combined = _filter_with_reason(combined, combined[target].notna(), f"{target}:present")
        log("rows.filter_target", column=target, rows_in=before, rows_out=int(combined.shape[0]))

    # --- bucket high-cardinality categoricals (only if configured) ---
    if config.get("bucketing_rules"):
        rules: Dict[str, int] = dict(config["bucketing_rules"])
        for c, threshold in rules.items():
            if c in combined.columns:
                combined[c] = _bucket_categorical(combined[c], int(threshold))
        log("categorical.bucket", rules=rules)

    # --- emit drop audit if requested ---
    audit_path = dropped_rows_path
    if audit_path is None and config.get("dropped_rows_path"):
        audit_path = Path(config["dropped_rows_path"])
    _emit_drop_audit(audit_path)

    return combined
