from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Mapping
import argparse
import pandas as pd
import numpy as np
import re
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
import warnings

from property_adviser.core.config import load_config
from property_adviser.core.io import (
    load_parquet_or_csv,
    save_parquet_or_csv,
    write_list,
    ensure_dir,
)
from property_adviser.core.app_logging import log, setup_logging
from property_adviser.feature.compute import compute_feature_scores_from_parquet

CONFIG_PATH = "config/features.yml"

ELIMINATION_ESTIMATORS = {
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
    scores_table: pd.DataFrame                    # full table: feature, pearson, mutual_info, eta, best_metric, best_score, selected, reason
    selected_columns: List[str]
    X: pd.DataFrame
    y: pd.Series | pd.DataFrame
    output_dir: Optional[Path] = None
    scores_path: Optional[Path] = None


def _normalise_mi_inplace(df: pd.DataFrame, col: str = "mutual_info") -> None:
    if col not in df.columns:
        return
    mi = df[col].astype(float)
    finite = mi.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if finite.empty:
        df[col] = pd.NA
        return
    mi_min, mi_max = float(finite.min()), float(finite.max())
    if mi_max > mi_min:
        df[col] = (mi - mi_min) / (mi_max - mi_min)
    else:
        df[col] = 0.0

def _tidy_scores(scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = [{"feature": f, **{k: float(v) for k, v in m.items()}} for f, m in scores.items()]
    df = pd.DataFrame(rows)
    for col in ("pearson_abs", "mutual_info", "eta"):
        if col not in df.columns:
            df[col] = pd.NA
    _normalise_mi_inplace(df, "mutual_info")  # <-- add this line

    metric_cols = ["pearson_abs", "mutual_info", "eta"]
    df["best_score"] = df[metric_cols].max(axis=1, skipna=True)
    df["best_metric"] = df[metric_cols].idxmax(axis=1, skipna=True)
    
    return df



def _apply_overrides(
    features: List[str],
    include: List[str],
    exclude: List[str]
) -> List[str]:
    """
    Apply manual include/exclude, preserving order & uniqueness.
    Includes first, then drops excludes.
    """
    feat_set = list(dict.fromkeys(features))  # stable-unique
    # ensure includes are first if not present already
    for f in include:
        if f not in feat_set:
            feat_set.insert(0, f)
    # drop excludes
    feat_set = [f for f in feat_set if f not in set(exclude)]
    return feat_set

import numpy as np

def _mark_reason(scores_df, feat, reason, selected_flag=None):
    idx = scores_df.index[scores_df["feature"] == feat]
    if len(idx):
        if selected_flag is not None:
            scores_df.loc[idx, "selected"] = bool(selected_flag)
        scores_df.loc[idx, "reason"] = reason if scores_df.loc[idx, "reason"].isna().all() else scores_df.loc[idx, "reason"].astype(str) + "; " + reason

def _drop_id_like(df, candidate_cols, scores_df, cfg, include):
    if not cfg.get("id_like", {}).get("enable", True):
        return candidate_cols
    min_ratio = float(cfg["id_like"].get("min_unique_ratio", 0.98))
    patterns = [re.compile(p, flags=re.IGNORECASE) for p in cfg["id_like"].get("drop_regex", [])]
    n = len(df)
    keep = []
    for c in candidate_cols:
        if c in include:
            keep.append(c); continue
        unique_ratio = (df[c].nunique(dropna=True) / max(n, 1)) if c in df.columns else 0.0
        name_hit = any(p.search(c) for p in patterns)
        if unique_ratio >= min_ratio or name_hit:
            _mark_reason(scores_df, c, f"dropped: id-like (unique_ratio={unique_ratio:.3f})", selected_flag=False)
        else:
            keep.append(c)
    return keep

def _apply_family_rules(candidate_cols, scores_df, cfg, include):
    families = cfg.get("families", {})
    if not families:
        return candidate_cols
    keep_set = set(candidate_cols)
    for fam_name, fam_cfg in families.items():
        fam_cols = fam_cfg.get("columns", [])
        fam_keep = set(fam_cfg.get("keep", []))
        present = [c for c in fam_cols if c in keep_set]
        if len(present) <= 1:
            continue
        # protect manual include
        protected = [c for c in present if c in include]
        if protected:
            chosen = set(protected)
        elif fam_keep:
            chosen = fam_keep & keep_set
            if not chosen:
                # if preferred not present, keep the first present
                chosen = {present[0]}
        else:
            chosen = {present[0]}
        for c in present:
            if c not in chosen:
                keep_set.discard(c)
                _mark_reason(scores_df, c, f"dropped: family({fam_name}) -> keep {sorted(chosen)}", selected_flag=False)
        for c in chosen:
            _mark_reason(scores_df, c, f"kept: family({fam_name})", selected_flag=True)
    return [c for c in candidate_cols if c in keep_set]

def _prune_correlated(df, candidate_cols, scores_df, cfg, include, best_score_map):
    red_cfg = cfg.get("redundancy", {})
    if not red_cfg.get("enable", True):
        return candidate_cols
    thr = float(red_cfg.get("threshold", 0.95))
    prefer_keep = set(red_cfg.get("prefer_keep", []))

    #numeric-only for correlation
    cols = [c for c in candidate_cols if c in df.columns]
    num_cols = [
        c for c in cols
        if pd.api.types.is_numeric_dtype(df[c].dropna().infer_objects())
    ]
    
    if len(num_cols) <= 1:
        return candidate_cols

    corr = df[num_cols].corr().abs()
    to_drop = set()
    kept = set(num_cols)

    # iterate upper triangle
    for i, c1 in enumerate(num_cols):
        if c1 not in kept: continue
        for c2 in num_cols[i+1:]:
            if c2 not in kept: continue
            r = corr.loc[c1, c2]
            if np.isnan(r) or r < thr:
                continue
            # decide which to drop
            if c1 in include and c2 not in include:
                drop = c2
            elif c2 in include and c1 not in include:
                drop = c1
            elif c1 in prefer_keep and c2 not in prefer_keep:
                drop = c2
            elif c2 in prefer_keep and c1 not in prefer_keep:
                drop = c1
            else:
                # drop the lower best_score
                s1 = best_score_map.get(c1, -np.inf)
                s2 = best_score_map.get(c2, -np.inf)
                drop = c1 if s1 < s2 else c2
            if drop in kept:
                kept.remove(drop)
                to_drop.add((drop, (c1 if drop==c2 else c2), r))

    final = [c for c in candidate_cols if (c not in [d[0] for d in to_drop])]
    for dcol, keeper, r in to_drop:
        _mark_reason(scores_df, dcol, f"dropped: high correlation with {keeper} (r={r:.3f})", selected_flag=False)
        _mark_reason(scores_df, keeper, f"kept over {dcol} (r={r:.3f})", selected_flag=True)
    return final


def _build_elimination_estimator(cfg: Mapping[str, Any]):
    name = cfg.get("estimator", "RandomForestRegressor")
    if name not in ELIMINATION_ESTIMATORS:
        raise ValueError(
            f"Unsupported elimination estimator '{name}'. Available: {sorted(ELIMINATION_ESTIMATORS)}"
        )
    params = cfg.get("estimator_params") or {}
    return ELIMINATION_ESTIMATORS[name](**params)


def _run_recursive_elimination(
    df: pd.DataFrame,
    target_col: str,
    candidate_cols: List[str],
    *,
    scores_df: pd.DataFrame,
    cfg: Mapping[str, Any],
    include: List[str],
) -> List[str]:
    elim_cfg = cfg.get("elimination", {}) or {}
    if not elim_cfg.get("enable", False):
        return candidate_cols

    if len(candidate_cols) <= 1:
        log(
            "feature_selection.elimination_skipped",
            reason="too_few_features",
            total=len(candidate_cols),
        )
        return candidate_cols

    min_features_cfg = int(elim_cfg.get("min_features", 5))
    if len(candidate_cols) <= max(1, min_features_cfg):
        log(
            "feature_selection.elimination_skipped",
            reason="under_min_features",
            total=len(candidate_cols),
            min_features=min_features_cfg,
        )
        return candidate_cols

    X = df[candidate_cols].copy()
    y = df[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in candidate_cols if c not in num_cols]

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
                list(num_cols),
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
                list(cat_cols),
            )
        )

    if not transformers:
        log(
            "feature_selection.elimination_skipped",
            reason="no_transformers",
            total=len(candidate_cols),
        )
        return candidate_cols

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    log(
        "feature_selection.elimination_start",
        estimator=elim_cfg.get("estimator", "RandomForestRegressor"),
        total=len(candidate_cols),
        numeric=len(num_cols),
        categorical=len(cat_cols),
    )

    try:
        X_processed = preprocessor.fit_transform(X)
    except Exception as exc:
        log(
            "feature_selection.elimination_failed",
            error=str(exc),
            stage="preprocess",
        )
        return candidate_cols

    if X_processed is None or X_processed.size == 0:
        return candidate_cols

    X_matrix = np.asarray(X_processed)
    y_array = np.asarray(y)

    log(
        "feature_selection.elimination_preprocessed",
        samples=int(X_matrix.shape[0]),
        encoded_features=int(X_matrix.shape[1]),
    )

    feature_names = list(preprocessor.get_feature_names_out())
    base_names: List[str] = []
    if num_cols:
        base_names.extend(num_cols)
    if cat_cols:
        cat_transformer: Pipeline = preprocessor.named_transformers_["cat"]  # type: ignore[index]
        ohe: OneHotEncoder = cat_transformer.named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        base_names.extend(name.split("_", 1)[0] for name in cat_feature_names)

    encoded_to_base = dict(zip(feature_names, base_names))
    if len(encoded_to_base) != len(feature_names):
        encoded_to_base = {}
        for name in feature_names:
            stripped = name.split("__", 1)[-1]
            base = stripped.split("_", 1)[0]
            encoded_to_base[name] = base

    try:
        estimator = _build_elimination_estimator(elim_cfg)
    except Exception as exc:
        log("feature_selection.elimination_failed", error=str(exc), stage="estimator")
        return candidate_cols

    step = int(max(1, elim_cfg.get("step", 1)))
    min_features_to_select = int(max(1, min(min_features_cfg, len(candidate_cols) - 1)))
    scoring = elim_cfg.get("scoring", "r2")
    cv = int(elim_cfg.get("cv", 3))
    n_jobs = elim_cfg.get("n_jobs")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            rfecv = RFECV(
                estimator=estimator,
                step=step,
                min_features_to_select=min_features_to_select,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
            )
            log(
                "feature_selection.elimination_rfecv_fit",
                samples=int(X_matrix.shape[0]),
                encoded_features=int(X_matrix.shape[1]),
                cv=cv,
                step=step,
            )
            rfecv.fit(X_matrix, y_array)
        except Exception as exc:
            log("feature_selection.elimination_failed", error=str(exc), stage="rfecv")
            return candidate_cols

    support_mask = rfecv.support_
    ranking = rfecv.ranking_

    base_support: Dict[str, bool] = {col: False for col in candidate_cols}
    base_rank: Dict[str, int] = {}

    for encoded_name, keep_flag, rank in zip(feature_names, support_mask, ranking):
        base = encoded_to_base.get(encoded_name)
        if base not in base_support:
            continue
        if keep_flag:
            base_support[base] = True
        base_rank[base] = min(rank, base_rank.get(base, rank))

    # honour manual includes
    for col in include:
        if col in base_support:
            base_support[col] = True
            base_rank.setdefault(col, 1)

    # Ensure columns exist in score table
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

    final_selection = [c for c in candidate_cols if base_support.get(c)]

    if not final_selection:
        return candidate_cols

    best_score = None
    if hasattr(rfecv, "cv_results_"):
        cv_results = rfecv.cv_results_
        if "mean_test_score" in cv_results:
            best_score = float(np.max(cv_results["mean_test_score"]))
    elif hasattr(rfecv, "grid_scores_"):
        scores = np.array(rfecv.grid_scores_)
        if scores.size:
            best_score = float(np.max(scores))

    log(
        "feature_selection.elimination",
        estimator=elim_cfg.get("estimator", "RandomForestRegressor"),
        selected=len(final_selection),
        total=len(candidate_cols),
        best_score=best_score,
        step=step,
        cv=cv,
    )

    return final_selection

def run_feature_selection(
    cfg: Dict[str, Any],
    *,
    # GUI overrides (defaults are blanks for batch processing)
    include: Optional[List[str]] = None,          # manually force-include
    exclude: Optional[List[str]] = None,          # manually drop
    use_top_k: Optional[bool] = None,             # None => follow config presence of top_k
    top_k: Optional[int] = None,                  # None => use cfg['top_k'] if present
    scores_output_filename: str = "feature_scores.parquet",
    write_outputs: bool = True
) -> FeatureSelectionResult:
    """
    Reusable entry point for CLI/GUI.

    cfg keys expected:
      - input_file (str), output_dir (str), target (str)
      - correlation_threshold (float)
      - exclude_columns (list[str])
      - mi_random_state (int)
      - top_k (optional int)
    """
    include = include or []
    exclude = exclude or []

    # 1) Compute scores and tidy
    raw_scores = compute_feature_scores_from_parquet(config=cfg)
    scores_df = _tidy_scores(raw_scores)          # builds pearson_abs/mutual_info/eta
    if "elimination_rank" not in scores_df.columns:
        scores_df["elimination_rank"] = pd.NA
    if "elimination_selected" not in scores_df.columns:
        scores_df["elimination_selected"] = pd.NA

    # 2) Load DF here so guardrails can use it
    df = load_parquet_or_csv(Path(cfg["input_file"]))
    target_col = cfg["target"]

    # 3) Initial selection: threshold OR top-k
    threshold = float(cfg["correlation_threshold"])
    # threshold mode from scores_df (now that MI is normalised)
    selected_cols = (
        scores_df.loc[scores_df["best_score"].fillna(-1) >= threshold, "feature"]
        .tolist()
    )

    cfg_top_k = cfg.get("top_k")
    effective_use_top_k = (use_top_k if use_top_k is not None else cfg_top_k is not None)
    effective_top_k = int(top_k if top_k is not None else (cfg_top_k or 0))

    if effective_use_top_k and effective_top_k > 0:
        ranked = scores_df.sort_values("best_score", ascending=False, na_position="last")
        selected_cols = ranked["feature"].head(effective_top_k).tolist()
        topk_rank_map = {f: i + 1 for i, f in enumerate(selected_cols)}
    else:
        topk_rank_map = {}

    # 4) Apply manual GUI overrides
    selected_cols = _apply_overrides(selected_cols, include=include, exclude=exclude)

    # 5) Init selection flags/reasons BEFORE guardrails
    if "reason" not in scores_df.columns:
        scores_df["reason"] = pd.NA
    scores_df["selected"] = scores_df["feature"].isin(set(selected_cols))

    # Manual include/exclude reasons
    for f in (include or []):
        _mark_reason(scores_df, f, "manual include", selected_flag=True)
    for f in (exclude or []):
        _mark_reason(scores_df, f, "manual exclude (not selected)", selected_flag=False)

    # Top-k reasons (if used)
    for f, rank in (topk_rank_map or {}).items():
        _mark_reason(scores_df, f, f"top_k rank {rank}/{len(topk_rank_map)}", selected_flag=True)
    
    # baseline flags before guardrails
    scores_df["selected"] = scores_df["feature"].isin(set(selected_cols))


    # Helper for redundancy tiebreaks
    best_score_map = dict(zip(scores_df["feature"], scores_df["best_score"]))

    # 6) Guardrails that REQUIRE DF
    selected_cols = _drop_id_like(df, selected_cols, scores_df, cfg, include or [])
    selected_cols = _apply_family_rules(selected_cols, scores_df, cfg, include or [])
    selected_cols = _prune_correlated(df, selected_cols, scores_df, cfg, include or [], best_score_map)
    selected_cols = _run_recursive_elimination(
        df,
        target_col,
        selected_cols,
        scores_df=scores_df,
        cfg=cfg,
        include=include or [],
    )

    # 7) Refresh final flags after guardrails
    scores_df["selected"] = scores_df["feature"].isin(set(selected_cols))

    # Build X / y (ensure we don’t include target by accident)
    selected_cols = [c for c in selected_cols if c in df.columns and c != target_col]
    X = df[selected_cols]
    y = df[target_col] if target_col in df else df[[target_col]]

    result = FeatureSelectionResult(
        scores_table=scores_df,
        selected_columns=selected_cols,
        X=X,
        y=y,
        output_dir=None,
        scores_path=None,
    )

    if write_outputs:
        out_dir = Path(ensure_dir(cfg["output_dir"]))
        result.output_dir = out_dir

        # Single “scores + selection” file (CSV or Parquet decided by extension)
        scores_path = out_dir / scores_output_filename
        save_parquet_or_csv(scores_df, scores_path)
        result.scores_path = scores_path

        # Selected list (nice to keep for debugging / pipelines)
        write_list(selected_cols, out_dir / "selected_features.txt")

        dataset_format = str(cfg.get("dataset_format", "csv")).lower()
        if dataset_format not in {"csv", "parquet"}:
            raise ValueError("dataset_format must be either 'csv' or 'parquet'")
        dataset_ext = ".parquet" if dataset_format == "parquet" else ".csv"

        # X / y outputs
        save_parquet_or_csv(X, out_dir / f"X{dataset_ext}")
        y_frame = y if isinstance(y, pd.DataFrame) else y.to_frame(name=target_col)
        save_parquet_or_csv(y_frame, out_dir / f"y{dataset_ext}")

        # Optional combined training set (kept for compatibility)
        train = df[selected_cols + [target_col]]
        save_parquet_or_csv(train, out_dir / f"training{dataset_ext}")

        log(
            "feature_selection.complete",
            features=len(selected_cols),
            output_dir=str(out_dir),
            threshold=threshold,
            use_top_k=effective_use_top_k,
            top_k=effective_top_k if effective_use_top_k else None,
            include=len(include),
            exclude=len(exclude),
        )

    return result


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Standalone feature selection.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to features.yml config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--scores-file",
        type=str,
        default="feature_scores.parquet",
        help="Name of the output file containing all features with scores and selection (parquet or csv).",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    cfg = load_config(Path(args.config))
    log("io.config_loaded", path=args.config)

    result = run_feature_selection(cfg, scores_output_filename=args.scores_file, write_outputs=True)

    print(f"Selected {len(result.selected_columns)} features → {result.output_dir}")
    print(f"Scores table → {result.scores_path}")
    return {
        "selected_columns": result.selected_columns,
        "n_selected": len(result.selected_columns),
        "output_dir": str(result.output_dir) if result.output_dir else None,
        "scores_path": str(result.scores_path) if result.scores_path else None,
    }


if __name__ == "__main__":
    main()
