from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from property_adviser.core.app_logging import setup_logging, log, log_exc, time_block  # type: ignore
from property_adviser.core.config import load_config

# -----------------------------------------------------------------------------
# Configuration structures
# -----------------------------------------------------------------------------

MODEL_FACTORY = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}


@dataclass
class Paths:
    X_path: Path
    y_path: Path
    feature_scores_path: Optional[Path]
    artifacts_dir: Path


@dataclass
class SplitCfg:
    validation_month: Optional[str]  # e.g. "2025-06" or "202506"
    month_column: str


@dataclass
class ModelSpec:
    enabled: bool
    grid: Dict[str, List[Any]]


@dataclass
class TrainCfg:
    task: str  # "regression" for now
    target: str  # e.g. "salePrice"
    paths: Paths
    split: SplitCfg
    models: Dict[str, ModelSpec]
    verbose: bool = False
    log_target: bool = False


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalise_month(s: pd.Series) -> pd.Series:
    """Accept 202506, '202506', '2025-06', '2025/06' -> '2025-06'."""
    def norm_one(v):
        if pd.isna(v):
            return np.nan
        v = str(v)
        if len(v) == 6 and v.isdigit():
            return f"{v[:4]}-{v[4:]}"
        v = v.replace("/", "-")
        if len(v) == 7 and v[4] == "-":
            return v
        # try to parse heuristically
        try:
            if len(v) == 8 and v.isdigit():
                return f"{v[:4]}-{v[4:6]}"
        except Exception:
            pass
        return v

    return s.map(norm_one)


def _infer_feature_sets(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _normalise_param_grid(grid: Optional[Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
    params: Dict[str, List[Any]] = {}
    if not grid:
        return params
    for key, values in grid.items():
        norm_key = key if key.startswith(("model__", "preprocessor__")) else f"model__{key}"
        params[norm_key] = values
    return params


def _prefix_if_log_target(grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    if not grid:
        return {}
    return {
        key if key.startswith("regressor__") else f"regressor__{key}": values
        for key, values in grid.items()
    }


def _clean_params(params: Dict[str, Any], log_target: bool) -> Dict[str, Any]:
    if not params:
        return {}
    if not log_target:
        return dict(params)
    cleaned: Dict[str, Any] = {}
    for key, value in params.items():
        cleaned[key.replace("regressor__", "", 1)] = value
    return cleaned


# -----------------------------------------------------------------------------
# Core training
# -----------------------------------------------------------------------------

def load_train_cfg(config_path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> TrainCfg:
    """
    Load YAML config (model.yml) and return a structured TrainCfg.
    Minimal, but robust to missing keys.
    """
    cfg_path = Path(config_path) if config_path else None
    cfg: Dict[str, Any] = {}
    if cfg_path and cfg_path.exists():
        cfg = load_config(cfg_path)
    if overrides:
        cfg = {**cfg, **overrides}

    input_cfg = cfg.get("input", {})
    base_path = Path(input_cfg.get("path", ".")).expanduser()

    def _as_path(value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate
        return base_path / candidate

    X_path = _as_path(input_cfg.get("X"))
    y_path = _as_path(input_cfg.get("y"))
    feature_scores_path = _as_path(input_cfg.get("feature_scores"))

    if X_path is None or y_path is None:
        raise ValueError("Config 'input' section must define 'X' and 'y' paths.")

    artifacts_cfg = cfg.get("model_path", {})
    artifacts_dir = Path(artifacts_cfg.get("base", "models")).expanduser()

    split_cfg_raw = cfg.get("split", {})
    split_cfg = SplitCfg(
        validation_month=split_cfg_raw.get("validation_month"),  # e.g. "2025-06"
        month_column=split_cfg_raw.get("month_column", "saleYearMonth"),
    )

    models_raw = cfg.get("models") or {name: {"enabled": True, "grid": {}} for name in MODEL_FACTORY}
    models = {
        name: ModelSpec(
            enabled=bool(entry.get("enabled", True)),
            grid=dict(entry.get("grid", {})) or {},
        )
        for name, entry in models_raw.items()
    }

    return TrainCfg(
        task=cfg.get("task", "regression"),
        target=cfg.get("target", "salePrice"),
        paths=Paths(
            X_path=X_path,
            y_path=y_path,
            feature_scores_path=feature_scores_path,
            artifacts_dir=artifacts_dir,
        ),
        split=split_cfg,
        models=models,
        verbose=bool(cfg.get("verbose", False)),
        log_target=bool(cfg.get("log_target", False)),
    )


def _apply_feature_overrides(X: pd.DataFrame, feature_scores: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    If feature_scores contains selected flags or include/exclude info, apply it.
    Expect columns like: feature, selected (bool) OR include/exclude markers.
    Falls back to leaving X untouched if nothing is found.
    """
    if feature_scores is None or feature_scores.empty:
        return X
    cols = feature_scores.columns.str.lower().tolist()
    feats_col = "feature" if "feature" in feature_scores.columns else None
    if not feats_col:
        return X
    fs = feature_scores.copy()
    fs.columns = fs.columns.str.lower()

    if "selected" in cols:
        keep = fs.loc[fs["selected"].astype(bool), "feature"].tolist()
        keep = [c for c in keep if c in X.columns]
        return X[keep] if keep else X

    if "include" in cols or "exclude" in cols:
        include = set(fs.loc[fs.get("include", pd.Series(dtype=bool)).astype(bool), "feature"].tolist())
        exclude = set(fs.loc[fs.get("exclude", pd.Series(dtype=bool)).astype(bool), "feature"].tolist())
        final = [c for c in X.columns if (not include or c in include) and c not in exclude]
        return X[final] if final else X

    return X


def _build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols, cat_cols = _infer_feature_sets(X)
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, num_cols, cat_cols


def _model_candidates(cfg: TrainCfg) -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    cand: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {}
    for name, spec in cfg.models.items():
        if not spec.enabled:
            continue
        if name not in MODEL_FACTORY:
            raise ValueError(
                f"Unsupported model '{name}'. Add it to MODEL_FACTORY or disable it in model.yml."
            )
        est = MODEL_FACTORY[name]()
        cand[name] = (est, _normalise_param_grid(spec.grid))
    if not cand:
        raise ValueError("No enabled models found. Check your model.yml 'models' section.")
    return cand


def _choose_validation_month(X: pd.DataFrame, month_col: str, requested: Optional[str]) -> str:
    months = _normalise_month(X[month_col])
    X = X.copy()
    X[month_col] = months
    available = sorted(m for m in months.dropna().unique())
    if not available:
        raise ValueError(f"No valid months in column '{month_col}'.")
    if requested:
        req = _normalise_month(pd.Series([requested])).iloc[0]
        if req not in available:
            raise ValueError(f"Requested validation_month={requested} not in available months: {available[-5:]}")
        return req
    return available[-1]  # default: last month


def _split_by_month(
    X: pd.DataFrame,
    y: pd.Series,
    month_col: str,
    val_month: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    months = _normalise_month(X[month_col])
    mask_train = months < val_month
    mask_val = months == val_month
    X_tr = X.loc[mask_train].drop(columns=[month_col], errors="ignore").copy()
    X_va = X.loc[mask_val].drop(columns=[month_col], errors="ignore").copy()
    y_tr = y.loc[mask_train].copy()
    y_va = y.loc[mask_val].copy()
    if X_tr.empty or X_va.empty:
        raise ValueError(f"Empty split: train={X_tr.shape}, val={X_va.shape}. Check month distribution.")
    return X_tr, X_va, y_tr, y_va


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def train_timeseries_model(config_path: Optional[str] = "model.yml",
                           overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main training function:
    - reads X, y (+ optional feature_scores)
    - selects a validation month (requested or last)
    - builds preprocessing + candidates
    - grid-searches each candidate
    - saves best model bundle + model scores CSV (both timestamped)
    - returns a dict consumable by GUI
    """
    setup_logging(overrides.get("verbose", False) if overrides else False)

    try:
        cfg = load_train_cfg(Path(config_path) if config_path else None, overrides)
        log("train.start",
            task=cfg.task,
            target=cfg.target,
            X=str(cfg.paths.X_path),
            y=str(cfg.paths.y_path),
            feature_scores=str(cfg.paths.feature_scores_path) if cfg.paths.feature_scores_path else None,
            artifacts=str(cfg.paths.artifacts_dir),
            month_column=cfg.split.month_column,
            requested_validation_month=cfg.split.validation_month,
            log_target=cfg.log_target,
        )

        # Load data
        X = _read_any(cfg.paths.X_path)
        y_df = _read_any(cfg.paths.y_path)
        if cfg.target not in y_df.columns:
            raise ValueError(f"Target '{cfg.target}' not found in y file columns: {list(y_df.columns)[:5]}")
        y = y_df[cfg.target]
        if cfg.split.month_column not in X.columns:
            raise ValueError(f"Month column '{cfg.split.month_column}' not found in X.")

        # Optional manual adjustments via feature_scores
        fs = None
        if cfg.paths.feature_scores_path and cfg.paths.feature_scores_path.exists():
            fs = _read_any(cfg.paths.feature_scores_path)
            log("train.feature_scores_loaded", rows=len(fs), cols=list(fs.columns))

        # Apply overrides (selected/include/exclude)
        X_effective = _apply_feature_overrides(X, fs)

        if cfg.log_target:
            mask_valid = y > 0
            if not mask_valid.all():
                dropped = int((~mask_valid).sum())
                X_effective = X_effective.loc[mask_valid].copy()
                y = y.loc[mask_valid].copy()
                log("train.log_target_drop_nonpositive", dropped=dropped)

        # Choose validation month and split
        val_month = _choose_validation_month(X_effective, cfg.split.month_column, cfg.split.validation_month)
        log("train.validation_month", month=val_month)
        X_tr, X_va, y_tr, y_va = _split_by_month(X_effective, y, cfg.split.month_column, val_month)
        log("train.split", train_rows=len(X_tr), val_rows=len(X_va), n_features=X_tr.shape[1])

        # Build preprocessor using training features
        preprocessor, num_cols, cat_cols = _build_preprocessor(X_tr)
        log("train.preprocessor", num=len(num_cols), cat=len(cat_cols))

        # Model candidates + grids
        candidates = _model_candidates(cfg)
        results: List[Dict[str, Any]] = []

        # Ensure artifacts dir
        _ensure_dir(cfg.paths.artifacts_dir)
        ts = _timestamp()

        best_name = None
        best_score = -np.inf
        best_bundle = None
        best_params: Dict[str, Any] = {}

        # Try each model with its grid
        for name, (est, grid) in candidates.items():
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", est)])
            estimator = pipe
            param_grid = grid or {}
            if cfg.log_target:
                estimator = TransformedTargetRegressor(
                    regressor=pipe,
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=False,
                )
                param_grid = _prefix_if_log_target(param_grid)

            with time_block("train.gridsearch", model=name):
                gs = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    scoring="r2",  # select on R²; we’ll still report MAE/RMSE
                    cv=3,          # small CV inside training set; val month is still our final check
                    n_jobs=-1,
                    refit=True,
                    verbose=0,
                )
                gs.fit(X_tr, y_tr)

            # Evaluate on the validation month
            y_hat = gs.predict(X_va)
            metrics = _evaluate(y_va.values, y_hat)
            log("train.validation", model=name, **metrics, best_cv_score=float(getattr(gs, "best_score_", np.nan)))

            # Record
            best_params_current = _clean_params(gs.best_params_, cfg.log_target)
            results.append({
                "model": name,
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "val_r2": metrics["r2"],
                "best_params": best_params_current,
                "best_cv_score": float(getattr(gs, "best_score_", np.nan)),
            })

            # Track best by R² on validation month
            if metrics["r2"] > best_score:
                best_score = metrics["r2"]
                best_name = name
                best_bundle = gs.best_estimator_
                best_params = best_params_current

        if best_bundle is None:
            raise RuntimeError("No model was successfully trained/evaluated.")

        # Save artifacts (timestamped)
        model_path = cfg.paths.artifacts_dir / f"best_model_{best_name}_{ts}.joblib"
        joblib.dump({
            "model": best_bundle,
            "target": cfg.target,
            "month_column": cfg.split.month_column,
            "validation_month": val_month,
            "feature_num": num_cols,
            "feature_cat": cat_cols,
            "best_params": best_params,
            "models_tried": [r["model"] for r in results],
            "log_target": cfg.log_target,
        }, model_path)
        log("train.save_model", path=str(model_path), model=best_name, r2=best_score)

        # Save model scores CSV (timestamped)
        scores_df = pd.DataFrame(results).sort_values("val_r2", ascending=False)
        scores_path = cfg.paths.artifacts_dir / f"model_scores_{ts}.csv"
        scores_df.to_csv(scores_path, index=False)
        log("train.save_scores", path=str(scores_path), rows=len(scores_df))

        # Return a GUI-friendly payload
        return {
            "best_model": best_name,
            "best_model_path": str(model_path),
            "scores_path": str(scores_path),
            "validation_month": val_month,
            "scores": scores_df.to_dict(orient="records"),
        }

    except Exception as exc:
        log_exc("train.error", exc)
        raise
