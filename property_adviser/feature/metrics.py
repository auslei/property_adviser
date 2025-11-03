from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import statsmodels.api as sm

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
    for _, g in vals.groupby(cats, observed=False):
        n = float(len(g))
        if n == 0:
            continue
        d = float(g.mean() - overall)
        ss_between += n * (d ** 2)

    return float((ss_between / ss_total) ** 0.5)


def bic_improvement_numeric(x: pd.Series, y: pd.Series) -> float:
    """
    BIC improvement vs intercept-only model for a single numeric feature.
    Positive values indicate the feature improves model fit (lower BIC).
    Returns NaN when not computable.
    """
    m = x.notna() & y.notna()
    if not m.any():
        return np.nan
    xv = pd.to_numeric(x[m], errors="coerce").astype(float)
    yv = pd.to_numeric(y[m], errors="coerce").astype(float)
    valid = xv.notna() & yv.notna()
    if not valid.any():
        return np.nan
    xv = xv[valid]
    yv = yv[valid]
    try:
        X1 = sm.add_constant(xv.to_frame(name="x"), has_constant="add")
        model = sm.OLS(yv.values, X1.values).fit()
        bic1 = float(model.bic)
        X0 = np.ones((len(yv), 1))
        null_model = sm.OLS(yv.values, X0).fit()
        bic0 = float(null_model.bic)
        return bic0 - bic1
    except Exception:
        return np.nan


def bic_improvement_categorical(cat: pd.Series, y: pd.Series) -> float:
    """
    BIC improvement vs intercept-only model for a categorical feature.
    Encodes categories as k-1 dummies (one-hot, drop_first=True).
    Returns 0.0 when only one level is present (no information) and NaN when not computable.
    """
    m = cat.notna() & y.notna()
    if not m.any():
        return np.nan
    cats = cat[m].astype("category")
    yv = pd.to_numeric(y[m], errors="coerce").astype(float)
    if cats.nunique() < 2:
        return 0.0
    try:
        dummies = pd.get_dummies(cats, drop_first=True)
        if dummies.shape[1] == 0:
            return 0.0
        X1 = np.column_stack([np.ones(len(dummies)), dummies.values])
        model = sm.OLS(yv.values, X1).fit()
        bic1 = float(model.bic)
        X0 = np.ones((len(yv), 1))
        null_model = sm.OLS(yv.values, X0).fit()
        bic0 = float(null_model.bic)
        return bic0 - bic1
    except Exception:
        return np.nan
