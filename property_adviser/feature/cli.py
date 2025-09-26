from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Mapping
import argparse
import json
import pandas as pd

from property_adviser.core.config import load_config
from property_adviser.core.io import (
    load_parquet_or_csv,
    save_parquet_or_csv,
    write_list,
    ensure_dir,
)
from property_adviser.core.app_logging import log, setup_logging

from property_adviser.feature.compute import compute_feature_scores_from_parquet
from property_adviser.feature.selector import select_top_features

CONFIG_PATH = "config/features.yml"


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

    # 1) Compute scores (uses cfg['input_file'] etc.)
    raw_scores = compute_feature_scores_from_parquet(config=cfg)
    scores_df = _tidy_scores(raw_scores)

    # 2) Threshold selection via selector (this already picks "best" metric per feature)
    threshold = float(cfg["correlation_threshold"])
    # After scores_df is built (and MI normalised), select by best_score
    metric_cols = ["pearson_abs", "mutual_info", "eta"]
    scores_df["best_score"] = scores_df[metric_cols].max(axis=1, skipna=True)
    scores_df["best_metric"] = scores_df[metric_cols].idxmax(axis=1, skipna=True)

    selected_cols_threshold = (
        scores_df.loc[scores_df["best_score"].fillna(-1) >= threshold, "feature"]
        .tolist()
    )
    # 3) Optional top_k decision
    cfg_top_k = cfg.get("top_k")
    if use_top_k is None:
        # follow config: use top_k if present
        effective_use_top_k = cfg_top_k is not None
    else:
        effective_use_top_k = bool(use_top_k)

    effective_top_k = int(top_k if top_k is not None else (cfg_top_k if cfg_top_k is not None else 0))

    if effective_use_top_k and effective_top_k > 0:
        # rank by best_score
        ranked = scores_df.sort_values("best_score", ascending=False, na_position="last")
        topk_cols = ranked["feature"].head(effective_top_k).tolist()
        selected_cols = topk_cols
        topk_rank_map = {f: i + 1 for i, f in enumerate(topk_cols)}
    else:
        selected_cols = selected_cols_threshold
        topk_rank_map = {}

    # 4) Apply manual GUI overrides
    selected_cols = _apply_overrides(selected_cols, include=include, exclude=exclude)

    # 5) Build selection flags & reasons on the full table
    sel_set = set(selected_cols)
    scores_df["selected"] = scores_df["feature"].isin(sel_set)

    def _reason(row) -> str:
        f = row["feature"]
        if f in include:
            return "manual include"
        if f in exclude:
            return "manual exclude (not selected)"
        if f in sel_set:
            if f in topk_rank_map:
                return f"top_k rank {topk_rank_map[f]}/{effective_top_k}"
            # threshold reason by best metric
            m = row["best_metric"]
            s = row["best_score"]
            if pd.isna(s) or pd.isna(m):
                return "selected (no score reason available)"
            return f"{m} {s:.4f} >= {threshold}"
        # not selected
        return ""
    scores_df["reason"] = scores_df.apply(_reason, axis=1)

    # 6) X / y for downstream steps (and GUI use)
    df = load_parquet_or_csv(Path(cfg["input_file"]))
    target_col = cfg["target"]

    # Keep only features that exist in the dataframe
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

        # X / y outputs
        save_parquet_or_csv(X, out_dir / "X.parquet")
        y_frame = y if isinstance(y, pd.DataFrame) else y.to_frame(name=target_col)
        save_parquet_or_csv(y_frame, out_dir / "y.parquet")

        # Optional combined training set (kept for compatibility)
        train = df[selected_cols + [target_col]]
        save_parquet_or_csv(train, out_dir / "training.parquet")

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
