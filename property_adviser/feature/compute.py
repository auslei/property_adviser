from __future__ import annotations
from typing import Dict, Mapping
from pathlib import Path
import pandas as pd

from property_adviser.common.io import load_parquet_or_csv
from property_adviser.feature_selection_util.metrics import pearson_abs, mutual_info_numeric, correlation_ratio


def compute_feature_scores(df, target, exclude, mi_rs):
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in dataset")

    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        raise TypeError(f"Target '{target}' must be numeric")
    
    candidates = [c for c in df.columns if c not in exclude | {target}]
    scores: Dict[str, Dict[str, float]] = {}
    
    # Numeric features → MI + Pearson
    num_cols = df[candidates].select_dtypes(include=["number"]).columns
    for col in num_cols:
        mi = mutual_info_numeric(df[col], y, random_state=mi_rs)
        pr = pearson_abs(df[col], y)
        d: Dict[str, float] = {}
        if pd.notna(mi):
            d["mutual_info"] = float(mi)
        if pd.notna(pr):
            d["pearson_abs"] = float(pr)
        if d:
            scores[col] = d
    
    # Categorical features → Correlation Ratio (η)
    cat_cols = df[candidates].select_dtypes(include=["object", "category", "string"]).columns
    for col in cat_cols:
        eta = correlation_ratio(df[col], y)
        if pd.notna(eta):
            scores[col] = {"eta": float(eta)}
            
    return scores
    
def compute_feature_scores_from_parquet(*, config: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    """
    Compute per-feature scores vs target.

    Required config keys:
      - input_file: str
      - target: str
      - exclude_columns: list[str]
      - mi_random_state: int
    """
    input_file = Path(config["input_file"])
    target = config["target"]
    exclude = set(config["exclude_columns"])
    mi_rs = int(config["mi_random_state"])

    df = load_parquet_or_csv(input_file)
    
    return compute_feature_scores(df, target=target, exclude = exclude, mi_rs = mi_rs)