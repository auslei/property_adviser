from __future__ import annotations
from typing import Dict, List

from src.common.config import load_config
from src.common.io import load_parquet_or_csv, save_parquet_or_csv, write_list, ensure_dir
from src.common.app_logging import log

from src.feature_selection_util.compute import compute_feature_scores_from_parquet
from src.feature_selection_util.selector import select_top_features


CONFIG_PATH = "config/features.yml"

from pathlib import Path

def main() -> Dict:
    """
    Orchestrates:
      - load config
      - compute feature scores (numeric: MI+Pearson, categorical: η)
      - select top features by threshold
      - save selected list + training dataset

    Required config keys:
      - input_file, output_dir, target,
      - correlation_threshold,
      - exclude_columns,
      - mi_random_state
    """
    cfg = load_config(Path(CONFIG_PATH))

    scores = compute_feature_scores_from_parquet(config=cfg)
    selected = select_top_features(scores, correlation_threshold=float(cfg["correlation_threshold"]))

    out_dir = ensure_dir(cfg["output_dir"])
    selected_cols = [list(d.keys())[0] for d in selected]

    # Save outputs
    write_list(selected_cols, f"{out_dir}/selected_features.txt")
    df = load_parquet_or_csv(cfg["input_file"])
    save_parquet_or_csv(df[selected_cols + [cfg["target"]]], f"{out_dir}/training.parquet")

    log("feature_selection.complete",
        features=len(selected_cols),
        output_dir=out_dir,
        threshold=float(cfg["correlation_threshold"]),
        target=cfg["target"])

    return {
        "selected": selected,                # [{col: {metric: score}}, ...]
        "selected_columns": selected_cols,   # [col, ...]
        "output_dir": out_dir
    }


if __name__ == "__main__":
    summary = main()
    print(f"Selected {len(summary['selected_columns'])} features → {summary['output_dir']}")