"""Feature selection orchestration with typed configuration."""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from property_adviser.core.app_logging import log
from property_adviser.core.io import ensure_dir, load_parquet_or_csv, save_parquet_or_csv, write_list
from property_adviser.feature.compute import compute_feature_scores
from property_adviser.feature.config import FeatureSelectionConfig

ELIMINATION_ESTIMATORS: Dict[str, Any] = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
}


@dataclass
class FeatureSelectionResult:
    scores_table: pd.DataFrame
    selected_columns: List[str]
    X: pd.DataFrame
    y: pd.Series | pd.DataFrame
    output_dir: Optional[Path]
    scores_path: Optional[Path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_mutual_info(scores: Dict[str, Dict[str, float]]) -> None:
    mi_vals = [d.get("mutual_info") for d in scores.values() if "mutual_info" in d]
    mi_vals = [v for v in mi_vals if v is not None]
    if not mi_vals:
        return
    mi_min, mi_max = min(mi_vals), max(mi_vals)
    if mi_max <= mi_min:
        for d in scores.values():
            if "mutual_info" in d:
                d["mutual_info"] = 0.0
        return
    span = mi_max - mi_min
    for d in scores.values():
        if "mutual_info" in d:
            d["mutual_info"] = float((d["mutual_info"] - mi_min) / span)


def _scores_dict_to_frame(scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for feature, metrics in scores.items():
        row = {"feature": feature}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("pearson_abs", "mutual_info", "eta"):
        if col not in df.columns:
            df[col] = pd.NA
    metric_cols = ["pearson_abs", "mutual_info", "eta"]
    df["best_score"] = df[metric_cols].max(axis=1, skipna=True)
    df["best_metric"] = df[metric_cols].idxmax(axis=1, skipna=True)
    return df


def _apply_overrides(features: List[str], include: Iterable[str], exclude: Iterable[str]) -> List[str]:
    ordered = list(dict.fromkeys(features))
    include = list(include or [])
    exclude_set = set(exclude or [])
    for inc in include:
        if inc not in ordered:
            ordered.insert(0, inc)
    ordered = [f for f in ordered if f not in exclude_set]
    return ordered


def _mark_reason(scores_df: pd.DataFrame, feat: str, reason: str, selected_flag: Optional[bool] = None) -> None:
    idx = scores_df.index[scores_df["feature"] == feat]
    if not len(idx):
        return
    if selected_flag is not None:
        scores_df.loc[idx, "selected"] = bool(selected_flag)
    current = scores_df.loc[idx, "reason"].astype(str)
    existing = scores_df.loc[idx, "reason"].where(scores_df.loc[idx, "reason"].notna(), "")
    append = existing.str.cat(pd.Series([reason] * len(idx), index=idx), sep="; ").str.strip("; ").replace({"": reason})
    scores_df.loc[idx, "reason"] = append


def _drop_id_like(
    df: pd.DataFrame,
    candidate_cols: List[str],
    scores_df: pd.DataFrame,
    *,
    min_unique_ratio: float,
    patterns: Iterable[str],
    include: Iterable[str],
) -> List[str]:
    keep: List[str] = []
    n = len(df)
    include_set = set(include)
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    for col in candidate_cols:
        if col in include_set:
            keep.append(col)
            continue
        if col not in df:
            continue
        unique_ratio = df[col].nunique(dropna=True) / max(n, 1)
        name_hit = any(p.search(col) for p in compiled)
        if unique_ratio >= min_unique_ratio or name_hit:
            _mark_reason(scores_df, col, f"dropped: id-like (ratio={unique_ratio:.3f})", selected_flag=False)
        else:
            keep.append(col)
    return keep


def _apply_family_rules(
    candidate_cols: List[str],
    scores_df: pd.DataFrame,
    families: Mapping[str, Any],
    include: Iterable[str],
) -> List[str]:
    if not families:
        return candidate_cols
    include_set = set(include)
    keep_set = set(candidate_cols)
    for family_name, family_cfg in families.items():
        columns = list(family_cfg.get("columns", []))
        present = [c for c in columns if c in keep_set]
        if len(present) <= 1:
            continue
        keep_pref = set(family_cfg.get("keep", []))
        if include_set.intersection(present):
            chosen = include_set.intersection(present)
        elif keep_pref.intersection(present):
            chosen = keep_pref.intersection(present)
        else:
            chosen = {present[0]}
        for col in present:
            if col not in chosen:
                keep_set.discard(col)
                _mark_reason(scores_df, col, f"dropped: family({family_name})", selected_flag=False)
        for col in chosen:
            _mark_reason(scores_df, col, f"kept: family({family_name})", selected_flag=True)
    return [c for c in candidate_cols if c in keep_set]


def _prune_correlated(
    df: pd.DataFrame,
    candidate_cols: List[str],
    scores_df: pd.DataFrame,
    *,
    threshold: float,
    prefer_keep: Iterable[str],
    include: Iterable[str],
    best_score_map: Mapping[str, Any],
) -> List[str]:
    cols = [c for c in candidate_cols if c in df.columns]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c].dropna().infer_objects())]
    if len(numeric_cols) <= 1:
        return candidate_cols

    corr = df[numeric_cols].corr().abs()
    include_set = set(include)
    prefer_keep_set = set(prefer_keep)
    kept = set(numeric_cols)
    dropped_pairs: List[tuple[str, str, float]] = []

    for i, c1 in enumerate(numeric_cols):
        if c1 not in kept:
            continue
        for c2 in numeric_cols[i + 1 :]:
            if c2 not in kept:
                continue
            r = corr.loc[c1, c2]
            if pd.isna(r) or r < threshold:
                continue
            if c1 in include_set and c2 not in include_set:
                drop = c2
            elif c2 in include_set and c1 not in include_set:
                drop = c1
            elif c1 in prefer_keep_set and c2 not in prefer_keep_set:
                drop = c2
            elif c2 in prefer_keep_set and c1 not in prefer_keep_set:
                drop = c1
            else:
                s1 = best_score_map.get(c1, -np.inf)
                s2 = best_score_map.get(c2, -np.inf)
                drop = c1 if s1 < s2 else c2
            if drop in kept:
                kept.remove(drop)
                dropped_pairs.append((drop, c1 if drop == c2 else c2, float(r)))

    for dropped, keeper, corr_val in dropped_pairs:
        _mark_reason(scores_df, dropped, f"dropped: high correlation with {keeper} (r={corr_val:.3f})", selected_flag=False)
        _mark_reason(scores_df, keeper, f"kept over {dropped} (r={corr_val:.3f})", selected_flag=True)

    return [c for c in candidate_cols if c in kept]


def _build_elimination_estimator(name: str, params: Mapping[str, Any]) -> Any:
    if name not in ELIMINATION_ESTIMATORS:
        raise ValueError(
            f"Unsupported elimination estimator '{name}'. Available: {sorted(ELIMINATION_ESTIMATORS)}"
        )
    estimator_cls = ELIMINATION_ESTIMATORS[name]
    return estimator_cls(**dict(params))


def _sample_for_elimination(df: pd.DataFrame, sample_rows: Optional[int], random_state: Optional[int]) -> pd.DataFrame:
    if sample_rows is None or sample_rows <= 0 or len(df) <= sample_rows:
        return df
    return df.sample(n=sample_rows, random_state=random_state)


def _run_recursive_elimination(
    df: pd.DataFrame,
    target_col: str,
    candidate_cols: List[str],
    *,
    scores_df: pd.DataFrame,
    config: FeatureSelectionConfig,
    include: Iterable[str],
    best_score_map: Mapping[str, Any],
) -> List[str]:
    elim_cfg = config.elimination
    if not elim_cfg.enable:
        return candidate_cols

    if len(candidate_cols) <= 1:
        log("feature_selection.elimination_skipped", reason="too_few_features", total=len(candidate_cols))
        return candidate_cols

    min_features = max(1, elim_cfg.min_features)
    if len(candidate_cols) <= min_features:
        log(
            "feature_selection.elimination_skipped",
            reason="under_min_features",
            total=len(candidate_cols),
            min_features=min_features,
        )
        return candidate_cols

    log("feature_selection.elimination_prepare", total=len(candidate_cols))
    df_for_elimination = _sample_for_elimination(df, elim_cfg.sample_rows, elim_cfg.random_state)

    columns_for_elimination = list(candidate_cols)
    max_features = elim_cfg.max_features
    if max_features is not None and max_features > 0 and len(columns_for_elimination) > max_features:
        sorted_cols = sorted(columns_for_elimination, key=lambda c: best_score_map.get(c, -np.inf), reverse=True)
        manually_included = [c for c in include if c in columns_for_elimination]
        top_cols = sorted_cols[:max_features]
        columns_for_elimination = list(dict.fromkeys(manually_included + top_cols))
        log(
            "feature_selection.elimination_subset",
            requested=max_features,
            actual=len(columns_for_elimination),
        )

    X = df_for_elimination[columns_for_elimination].copy()
    y = df_for_elimination[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in columns_for_elimination if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            )
        )

    if not transformers:
        log("feature_selection.elimination_skipped", reason="no_transformers", total=len(columns_for_elimination))
        return candidate_cols

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    try:
        estimator = _build_elimination_estimator(elim_cfg.estimator, elim_cfg.estimator_params)
    except Exception as exc:
        log("feature_selection.elimination_failed", stage="estimator", error=str(exc))
        return candidate_cols

    start = perf_counter()
    try:
        X_processed = preprocessor.fit_transform(X)
    except Exception as exc:
        log("feature_selection.elimination_failed", stage="preprocess", error=str(exc))
        return candidate_cols

    elapsed_pre = perf_counter() - start
    log(
        "feature_selection.elimination_preprocessed",
        samples=int(X_processed.shape[0]),
        encoded_features=int(X_processed.shape[1]),
        duration=round(elapsed_pre, 3),
    )

    rfecv = RFECV(
        estimator=estimator,
        step=max(1, elim_cfg.step),
        min_features_to_select=max(1, min(elim_cfg.min_features, len(columns_for_elimination) - 1)),
        scoring=elim_cfg.scoring,
        cv=max(2, elim_cfg.cv),
        n_jobs=elim_cfg.n_jobs,
    )

    start_fit = perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            rfecv.fit(np.asarray(X_processed), np.asarray(y))
        except Exception as exc:
            log("feature_selection.elimination_failed", stage="rfecv", error=str(exc))
            return candidate_cols
    elapsed_fit = perf_counter() - start_fit

    feature_names = list(preprocessor.get_feature_names_out())
    base_names: List[str] = []
    base_names.extend(num_cols)
    if cat_cols:
        cat_transformer: Pipeline = preprocessor.named_transformers_["cat"]  # type: ignore[index]
        ohe: OneHotEncoder = cat_transformer.named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        base_names.extend(name.split("_", 1)[0] for name in cat_feature_names)
    encoded_to_base = dict(zip(feature_names, base_names))

    support_mask = rfecv.support_
    ranking = rfecv.ranking_

    base_support: Dict[str, bool] = {col: False for col in columns_for_elimination}
    base_rank: Dict[str, int] = {}
    for encoded_name, keep_flag, rank in zip(feature_names, support_mask, ranking):
        base = encoded_to_base.get(encoded_name)
        if base not in base_support:
            continue
        if keep_flag:
            base_support[base] = True
        base_rank[base] = min(rank, base_rank.get(base, rank))

    for inc in include:
        if inc in base_support:
            base_support[inc] = True
            base_rank.setdefault(inc, 1)

    if "elimination_rank" not in scores_df.columns:
        scores_df["elimination_rank"] = pd.NA
    if "elimination_selected" not in scores_df.columns:
        scores_df["elimination_selected"] = pd.NA

    for col, selected in base_support.items():
        idx = scores_df.index[scores_df["feature"] == col]
        if len(idx):
            if col in base_rank:
                scores_df.loc[idx, "elimination_rank"] = int(base_rank[col])
            scores_df.loc[idx, "elimination_selected"] = bool(selected)
        if selected:
            _mark_reason(scores_df, col, "kept: elimination", selected_flag=True)
        else:
            _mark_reason(scores_df, col, "dropped: elimination", selected_flag=False)

    selected_after_elim = {c for c, keep in base_support.items() if keep}
    if not selected_after_elim:
        return candidate_cols

    retained: List[str] = []
    subset = set(columns_for_elimination)
    for col in candidate_cols:
        if col in subset:
            if col in selected_after_elim:
                retained.append(col)
        else:
            retained.append(col)

    best_score = None
    if hasattr(rfecv, "cv_results_") and "mean_test_score" in rfecv.cv_results_:
        best_score = float(np.max(rfecv.cv_results_["mean_test_score"]))
    elif hasattr(rfecv, "grid_scores_"):
        scores = np.asarray(rfecv.grid_scores_)
        if scores.size:
            best_score = float(np.max(scores))

    log(
        "feature_selection.elimination_complete",
        estimator=elim_cfg.estimator,
        selected=len(selected_after_elim),
        total=len(columns_for_elimination),
        duration=round(elapsed_pre + elapsed_fit, 3),
        fit_duration=round(elapsed_fit, 3),
        best_score=best_score,
        step=max(1, elim_cfg.step),
        cv=max(2, elim_cfg.cv),
    )

    return retained


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_feature_selection(
    config: FeatureSelectionConfig,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    use_top_k: Optional[bool] = None,
    top_k: Optional[int] = None,
    write_outputs: bool = True,
    scores_output_filename: Optional[str] = None,
) -> FeatureSelectionResult:
    overall_start = perf_counter()

    include = list(include or [])
    exclude = list(exclude or [])

    df = load_parquet_or_csv(config.input_file)
    if config.target not in df.columns:
        raise KeyError(f"Target column '{config.target}' not found in dataset")

    log(
        "feature_selection.start",
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        target=config.target,
        input=str(config.input_file),
    )

    score_start = perf_counter()
    scores = compute_feature_scores(
        df,
        target=config.target,
        exclude=set(config.exclude_columns),
        mi_rs=config.mi_random_state,
    )
    _normalise_mutual_info(scores)
    score_elapsed = perf_counter() - score_start

    scores_df = _scores_dict_to_frame(scores)
    if "reason" not in scores_df.columns:
        scores_df["reason"] = pd.NA

    threshold = config.correlation_threshold
    effective_use_top_k = use_top_k if use_top_k is not None else config.use_top_k
    effective_top_k = top_k if top_k is not None else config.top_k

    selected_cols: List[str]
    topk_rank_map: Dict[str, int] = {}
    if effective_use_top_k and effective_top_k:
        sorted_df = scores_df.sort_values("best_score", ascending=False)
        selected_cols = sorted_df.head(effective_top_k)["feature"].tolist()
        topk_rank_map = {
            feat: idx + 1
            for idx, feat in enumerate(selected_cols)
        }
        log(
            "feature_selection.selection_mode",
            mode="top_k",
            top_k=effective_top_k,
        )
    else:
        if threshold is None:
            raise ValueError("correlation_threshold must be provided when top-k mode is not used")
        selected_cols = scores_df.loc[scores_df["best_score"] >= threshold, "feature"].tolist()
        log(
            "feature_selection.selection_mode",
            mode="threshold",
            threshold=threshold,
        )

    selected_cols = _apply_overrides(selected_cols, include=include, exclude=exclude)
    scores_df["selected"] = scores_df["feature"].isin(set(selected_cols))

    for f in include:
        _mark_reason(scores_df, f, "manual include", selected_flag=True)
    for f in exclude:
        _mark_reason(scores_df, f, "manual exclude (not selected)", selected_flag=False)
    for feat, rank in topk_rank_map.items():
        _mark_reason(scores_df, feat, f"top_k rank {rank}/{len(topk_rank_map)}", selected_flag=True)

    best_score_map = dict(zip(scores_df["feature"], scores_df["best_score"]))

    if config.id_like.enable:
        selected_cols = _drop_id_like(
            df,
            selected_cols,
            scores_df,
            min_unique_ratio=config.id_like.min_unique_ratio,
            patterns=config.id_like.drop_regex,
            include=include,
        )
    selected_cols = _apply_family_rules(selected_cols, scores_df, config.families, include)
    if config.redundancy.enable:
        selected_cols = _prune_correlated(
            df,
            selected_cols,
            scores_df,
            threshold=config.redundancy.threshold,
            prefer_keep=config.redundancy.prefer_keep,
            include=include,
            best_score_map=best_score_map,
        )
    selected_cols = _run_recursive_elimination(
        df,
        config.target,
        selected_cols,
        scores_df=scores_df,
        config=config,
        include=include,
        best_score_map=best_score_map,
    )

    selected_cols = [c for c in selected_cols if c in df.columns and c != config.target]
    scores_df["selected"] = scores_df["feature"].isin(set(selected_cols))

    X = df[selected_cols]
    y = df[config.target] if config.target in df else df[[config.target]]

    output_dir: Optional[Path] = None
    scores_path: Optional[Path] = None

    if write_outputs:
        out_dir = ensure_dir(config.output_dir)
        output_dir = Path(out_dir)
        scores_filename = scores_output_filename or config.scores_filename
        scores_path = output_dir / scores_filename
        save_parquet_or_csv(scores_df, scores_path)

        write_list(selected_cols, output_dir / "selected_features.txt")

        dataset_ext = ".parquet" if config.dataset_format == "parquet" else ".csv"
        save_parquet_or_csv(X, output_dir / f"X{dataset_ext}")
        y_frame = y if isinstance(y, pd.DataFrame) else y.to_frame(name=config.target)
        save_parquet_or_csv(y_frame, output_dir / f"y{dataset_ext}")
        train = df[selected_cols + [config.target]]
        save_parquet_or_csv(train, output_dir / f"training{dataset_ext}")

    elapsed = perf_counter() - overall_start
    log(
        "feature_selection.complete",
        selected=len(selected_cols),
        total_features=len(scores_df),
        duration=round(elapsed, 3),
        scoring_duration=round(score_elapsed, 3),
        output_dir=str(output_dir) if output_dir else None,
    )

    return FeatureSelectionResult(
        scores_table=scores_df,
        selected_columns=selected_cols,
        X=X,
        y=y,
        output_dir=output_dir,
        scores_path=scores_path,
    )


__all__ = ["FeatureSelectionResult", "run_feature_selection"]
