from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def pearson_abs(x: pd.Series, y: pd.Series) -> float:
    """Absolute Pearson correlation for two numeric series; drops NaNs pairwise."""
    m = x.notna() & y.notna()
    if not m.any():
        return np.nan
    r = x[m].corr(y[m])
    return float(abs(r)) if pd.notna(r) else np.nan

def mutual_info_numeric(x: pd.Series, y: pd.Series, *, random_state: int) -> float:
    """
    Mutual information (regression) for a single numeric feature vs numeric target.
    Captures non-linear dependence. Drops NaNs pairwise.
    """
    m = x.notna() & y.notna()
    if not m.any():
        return np.nan
    X = x[m].astype(float).to_numpy().reshape(-1, 1)
    Y = y[m].astype(float).to_numpy()
    mi = mutual_info_regression(X, Y, random_state=random_state)
    return float(mi[0]) if len(mi) else np.nan

def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """
    Correlation Ratio (η) between categorical feature and numeric target.
    No encoding or ordering assumed. Returns η in [0, 1].
    """
    m = categories.notna() & values.notna()
    if not m.any():
        return np.nan
    cats = categories[m].astype("category")
    vals = values[m].astype(float)

    overall = float(vals.mean())
    ss_total = float(((vals - overall) ** 2).sum())
    if ss_total == 0.0:
        return 0.0

    ss_between = 0.0
    for _, g in vals.groupby(cats):
        n = float(len(g))
        if n == 0:
            continue
        d = float(g.mean() - overall)
        ss_between += n * (d ** 2)

    return float((ss_between / ss_total) ** 0.5)