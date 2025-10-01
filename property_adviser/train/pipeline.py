"""Training pipeline orchestration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

from property_adviser.core.app_logging import log, time_block
from property_adviser.core.io import ensure_dir, load_parquet_or_csv, save_parquet_or_csv, write_list
from property_adviser.train.config import TrainingConfig

MODEL_FACTORY = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}


@dataclass
class TrainingResult:
    name: str
    target: str
    timestamp: str
    best_model: str
    best_model_path: Path
    canonical_model_path: Path
    summary_path: Path
    scores_path: Path
    metadata_path: Path
    validation_month: str
    scores_table: pd.DataFrame
    duration: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_month(series: pd.Series) -> pd.Series:
    def norm_one(value: Any) -> Optional[str]:
        if pd.isna(value):
            return None
        v = str(value)
        if len(v) == 6 and v.isdigit():
            return f"{v[:4]}-{v[4:]}"
        v = v.replace("/", "-")
        if len(v) == 7 and v[4] == "-":
            return v
        if len(v) == 8 and v.isdigit():
            return f"{v[:4]}-{v[4:6]}"
        return v

    return series.map(norm_one)


def _infer_feature_sets(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def _timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalise_param_grid(grid: Mapping[str, Any]) -> Dict[str, List[Any]]:
    params: Dict[str, List[Any]] = {}
    for key, values in grid.items():
        norm_key = key if key.startswith(("model__", "preprocessor__")) else f"model__{key}"
        params[norm_key] = list(values if isinstance(values, (list, tuple)) else [values])
    return params


def _prefix_if_log_target(grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    return {
        (key if key.startswith("regressor__") else f"regressor__{key}"): values
        for key, values in grid.items()
    }


def _clean_params(params: Mapping[str, Any], log_target: bool) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in params.items():
        if log_target and key.startswith("regressor__"):
            cleaned[key.replace("regressor__", "", 1)] = value
        elif log_target and key.startswith("model__"):
            cleaned[key.replace("model__", "", 1)] = value
        elif key.startswith("model__"):
            cleaned[key.replace("model__", "", 1)] = value
        else:
            cleaned[key] = value
    return cleaned


def _build_feature_metadata(
    *,
    X_full: pd.DataFrame,
    month_column: str,
    numeric_features: List[str],
    categorical_features: List[str],
    model_input_columns: List[str],
    timestamp: str,
) -> Dict[str, Any]:
    def _median(series: pd.Series) -> Optional[float]:
        numeric = pd.to_numeric(series, errors="coerce")
        value = numeric.median(skipna=True)
        if pd.isna(value):
            return None
        return float(value)

    def _mode(series: pd.Series) -> Optional[str]:
        clean = series.dropna()
        if clean.empty:
            return None
        mode_series = clean.mode(dropna=True)
        if mode_series.empty:
            return None
        return str(mode_series.iloc[0])

    numeric_impute = {col: _median(X_full[col]) if col in X_full.columns else None for col in numeric_features}
    categorical_impute = {col: _mode(X_full[col]) if col in X_full.columns else None for col in categorical_features}

    for col, value in list(numeric_impute.items()):
        if value is None:
            numeric_impute[col] = 0.0
    for col, value in list(categorical_impute.items()):
        if value is None:
            categorical_impute[col] = "Unknown"

    return {
        "timestamp": timestamp,
        "month_column": month_column,
        "raw_feature_columns": sorted(X_full.columns.tolist()),
        "model_input_columns": model_input_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "impute": {"numeric": numeric_impute, "categorical": categorical_impute},
        "property_age": {
            "bands": [5, 20],
            "labels": ["0-5", "6-20", "21+"],
        },
    }


def _apply_feature_overrides(X: pd.DataFrame, feature_scores: Optional[pd.DataFrame]) -> pd.DataFrame:
    if feature_scores is None or feature_scores.empty:
        return X
    fs = feature_scores.copy()
    fs.columns = fs.columns.str.lower()
    if "feature" not in fs.columns:
        return X
    features = fs["feature"].astype(str)
    if "selected" in fs.columns:
        keep = features[fs["selected"].astype(bool)].tolist()
        keep = [c for c in keep if c in X.columns]
        return X[keep] if keep else X
    include = set(features[fs.get("include", pd.Series(dtype=bool)).astype(bool)])
    exclude = set(features[fs.get("exclude", pd.Series(dtype=bool)).astype(bool)])
    if include:
        keep_cols = [c for c in X.columns if c in include and c not in exclude]
    else:
        keep_cols = [c for c in X.columns if c not in exclude]
    return X[keep_cols] if keep_cols else X


def _build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols, cat_cols = _infer_feature_sets(X)
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, num_cols, cat_cols


def _model_candidates(models_cfg: Mapping[str, Any]) -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    candidates: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {}
    for name, model_cfg in models_cfg.items():
        if not model_cfg.enabled:
            continue
        if name not in MODEL_FACTORY:
            raise ValueError(f"Unsupported model '{name}'.")
        estimator = MODEL_FACTORY[name]()
        param_grid = _normalise_param_grid(model_cfg.grid)
        candidates[name] = (estimator, param_grid)
    if not candidates:
        raise ValueError("No enabled models configured.")
    return candidates


def _choose_validation_month(X: pd.DataFrame, month_column: str, requested: Optional[str]) -> str:
    months = _normalise_month(X[month_column])
    available = sorted(m for m in months.dropna().unique())
    if not available:
        raise ValueError(f"No valid months found in column '{month_column}'.")
    if requested:
        req = _normalise_month(pd.Series([requested])).iloc[0]
        if req not in available:
            raise ValueError(
                f"Requested validation_month={requested} not present. Available months: {available[-5:]}"
            )
        return req
    return available[-1]


def _split_by_month(
    X: pd.DataFrame,
    y: pd.Series,
    month_column: str,
    validation_month: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    months = _normalise_month(X[month_column])
    mask_train = months < validation_month
    mask_val = months == validation_month

    train_idx = months[mask_train].sort_values().index
    val_idx = months[mask_val].sort_values().index

    X_tr = X.loc[train_idx].drop(columns=[month_column], errors="ignore").copy()
    X_va = X.loc[val_idx].drop(columns=[month_column], errors="ignore").copy()
    y_tr = y.loc[train_idx].copy()
    y_va = y.loc[val_idx].copy()
    if X_tr.empty or X_va.empty:
        raise ValueError(
            f"Received an empty split (train={X_tr.shape}, val={X_va.shape}). Check month distribution."
        )
    return X_tr, X_va, y_tr, y_va


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_training(config: TrainingConfig) -> TrainingResult:
    overall_start = perf_counter()

    X = load_parquet_or_csv(config.input.X)
    y_frame = load_parquet_or_csv(config.input.y)
    if config.target not in y_frame.columns:
        raise ValueError(f"Target '{config.target}' not found in y columns: {list(y_frame.columns)[:5]}")
    y = y_frame[config.target]
    if config.split.month_column not in X.columns:
        raise ValueError(f"Month column '{config.split.month_column}' not present in feature matrix.")

    feature_scores = None
    if config.input.feature_scores and Path(config.input.feature_scores).exists():
        feature_scores = load_parquet_or_csv(config.input.feature_scores)
        log(
            "train.feature_scores_loaded",
            rows=len(feature_scores),
            cols=list(feature_scores.columns),
            target=config.target,
            config_name=config.name,
        )

    X_effective = _apply_feature_overrides(X, feature_scores)

    if config.log_target:
        mask_valid = y > 0
        if not mask_valid.all():
            dropped = int((~mask_valid).sum())
            log(
                "train.log_target_drop_nonpositive",
                dropped=dropped,
                target=config.target,
                config_name=config.name,
            )
            X_effective = X_effective.loc[mask_valid].copy()
            y = y.loc[mask_valid].copy()

    validation_month = _choose_validation_month(
        X_effective, config.split.month_column, config.split.validation_month
    )
    log(
        "train.validation_month",
        month=validation_month,
        target=config.target,
        config_name=config.name,
    )

    X_tr, X_va, y_tr, y_va = _split_by_month(
        X_effective, y, config.split.month_column, validation_month
    )
    log(
        "train.split",
        train_rows=len(X_tr),
        val_rows=len(X_va),
        n_features=X_tr.shape[1],
        target=config.target,
        config_name=config.name,
    )

    n_splits = min(5, max(2, len(X_tr) // 120)) if len(X_tr) > 10 else 2
    cv_strategy = TimeSeriesSplit(n_splits=n_splits)
    log(
        "train.cv_setup",
        strategy="TimeSeriesSplit",
        n_splits=n_splits,
        train_rows=len(X_tr),
        target=config.target,
        config_name=config.name,
    )

    preprocessor, numeric_features, categorical_features = _build_preprocessor(X_tr)
    log(
        "train.preprocessor",
        num=len(numeric_features),
        cat=len(categorical_features),
        target=config.target,
        config_name=config.name,
    )

    candidates = _model_candidates(config.models)

    artifacts_dir = Path(ensure_dir(config.artifacts_dir))
    timestamp = _timestamp()

    metadata = _build_feature_metadata(
        X_full=X_effective,
        month_column=config.split.month_column,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_input_columns=X_tr.columns.tolist(),
        timestamp=timestamp,
    )

    results: List[Dict[str, Any]] = []
    best_model_name: Optional[str] = None
    best_model_score = -np.inf
    best_bundle = None
    best_params: Dict[str, Any] = {}

    log(
        "train.start",
        task=config.task,
        target=config.target,
        config_name=config.name,
        X=str(config.input.X),
        y=str(config.input.y),
        feature_scores=str(config.input.feature_scores) if config.input.feature_scores else None,
        artifacts=str(artifacts_dir),
        month_column=config.split.month_column,
        requested_validation_month=config.split.validation_month,
        log_target=config.log_target,
    )

    for name, (estimator, grid) in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        param_grid = dict(grid)
        trained_estimator: Any = pipeline
        if config.log_target:
            trained_estimator = TransformedTargetRegressor(
                regressor=pipeline,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )
            param_grid = _prefix_if_log_target(param_grid)

        with time_block("train.gridsearch", model=name, target=config.target, config_name=config.name):
            gs = GridSearchCV(
                estimator=trained_estimator,
                param_grid=param_grid,
                scoring="r2",
                cv=cv_strategy,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            gs.fit(X_tr, y_tr)

        y_pred = gs.predict(X_va)
        metrics = _evaluate(y_va.values, y_pred)
        log(
            "train.validation",
            model=name,
            target=config.target,
            config_name=config.name,
            **metrics,
            best_cv_score=float(getattr(gs, "best_score_", np.nan)),
        )

        cleaned_params = _clean_params(gs.best_params_, config.log_target)
        results.append(
            {
                "model": name,
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "val_r2": metrics["r2"],
                "best_params": cleaned_params,
                "best_cv_score": float(getattr(gs, "best_score_", np.nan)),
            }
        )

        if metrics["r2"] > best_model_score:
            best_model_score = metrics["r2"]
            best_model_name = name
            best_bundle = gs.best_estimator_
            best_params = cleaned_params

    if best_bundle is None or best_model_name is None:
        raise RuntimeError("No model was successfully trained.")

    bundle = {
        "model": best_bundle,
        "target": config.target,
        "month_column": config.split.month_column,
        "validation_month": validation_month,
        "feature_num": numeric_features,
        "feature_cat": categorical_features,
        "best_params": best_params,
        "models_tried": [r["model"] for r in results],
        "log_target": config.log_target,
    }

    model_path = artifacts_dir / f"best_model_{best_model_name}_{timestamp}.joblib"
    joblib.dump(bundle, model_path)
    log(
        "train.save_model",
        path=str(model_path),
        model=best_model_name,
        r2=best_model_score,
        target=config.target,
        config_name=config.name,
    )

    canonical_model_path = artifacts_dir / "best_model.joblib"
    joblib.dump(bundle, canonical_model_path)

    best_metrics = next((r for r in results if r["model"] == best_model_name), None)
    summary = {
        "model": best_model_name,
        "timestamp": timestamp,
        "validation_month": validation_month,
        "metrics": best_metrics or {},
        "best_params": best_params,
        "timestamped_model_path": str(model_path),
        "models_tried": [r["model"] for r in results],
        "log_target": config.log_target,
    }
    summary_path = artifacts_dir / "best_model.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(
        "train.save_model_canonical",
        model=str(canonical_model_path),
        summary=str(summary_path),
        target=config.target,
        config_name=config.name,
    )

    scores_df = pd.DataFrame(results).sort_values("val_r2", ascending=False)
    scores_path = artifacts_dir / f"model_scores_{timestamp}.csv"
    scores_df.to_csv(scores_path, index=False)
    log(
        "train.save_scores",
        path=str(scores_path),
        rows=len(scores_df),
        target=config.target,
        config_name=config.name,
    )

    metadata_payload = {
        "target": config.target,
        "validation_month": validation_month,
        "models_considered": [r["model"] for r in results],
        "selected_model": best_model_name,
        "feature_metadata": metadata,
    }
    metadata_path = config.input.base / "feature_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))
    log(
        "train.save_feature_metadata",
        path=str(metadata_path),
        target=config.target,
        config_name=config.name,
    )

    duration = perf_counter() - overall_start
    log(
        "train.complete",
        duration=round(duration, 3),
        best_model=best_model_name,
        validation_month=validation_month,
        target=config.target,
        config_name=config.name,
    )

    return TrainingResult(
        name=config.name,
        target=config.target,
        timestamp=timestamp,
        best_model=best_model_name,
        best_model_path=model_path,
        canonical_model_path=canonical_model_path,
        summary_path=summary_path,
        scores_path=scores_path,
        metadata_path=metadata_path,
        validation_month=validation_month,
        scores_table=scores_df,
        duration=duration,
    )


__all__ = ["MODEL_FACTORY", "TrainingResult", "run_training"]
