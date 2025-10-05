"""Convenience CLI for running the full Property Adviser pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from property_adviser.core.app_logging import log, log_exc, setup_logging
from property_adviser.core.config import load_config
from property_adviser.feature.config import load_feature_selection_config
from property_adviser.feature.cli import run_feature_selection
from property_adviser.preprocess import load_preprocess_config, run_preprocessing
from property_adviser.train.model_training import train_timeseries_model


def run_full_pipeline(
    *,
    preprocess_config: Path,
    feature_config: Path,
    model_config: Path,
    verbose: bool = False,
) -> List[Tuple[str, Path | None]]:
    """Execute preprocess → feature selection → training sequentially.

    Returns a list of (stage_name, output_path) tuples for quick inspection.
    Raises the first encountered exception after logging it.
    """
    setup_logging(verbose=verbose)

    stages: List[Tuple[str, Path | None]] = []

    try:
        pre_cfg = load_preprocess_config(preprocess_config)
        preprocess_result = run_preprocessing(pre_cfg, write_outputs=True)
        stages.append(("preprocess", preprocess_result.derived_path))
        log("pipeline.preprocess_complete", path=str(preprocess_result.derived_path))
    except Exception as exc:  # pragma: no cover - we want to bubble up
        log_exc("pipeline.preprocess_failed", exc)
        raise

    try:
        # Support multi-target feature selection configs
        fs_configs = load_feature_selection_config(feature_config)
        last_scores_path: Path | None = None
        total_selected = 0
        for cfg in fs_configs:
            fs_result = run_feature_selection(cfg, write_outputs=True)
            last_scores_path = fs_result.scores_path or last_scores_path
            total_selected += len(fs_result.selected_columns)
        stages.append(("feature_selection", last_scores_path))
        log(
            "pipeline.feature_selection_complete",
            scores=str(last_scores_path) if last_scores_path else None,
            targets=len(fs_configs),
            total_selected=total_selected,
        )
    except Exception as exc:  # pragma: no cover - we want to bubble up
        log_exc("pipeline.feature_selection_failed", exc)
        raise

    try:
        train_outcome = train_timeseries_model(
            config_path=str(model_config),
            overrides={"verbose": verbose},
        )
        stages.append(("train", Path(train_outcome["canonical_model_path"])))
        log(
            "pipeline.train_complete",
            model=train_outcome["best_model"],
            path=train_outcome["canonical_model_path"],
        )
    except Exception as exc:  # pragma: no cover - we want to bubble up
        log_exc("pipeline.train_failed", exc)
        raise

    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocess → feature → train in one go")
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=Path("config/preprocessing.yml"),
        help="Path to preprocessing.yml",
    )
    parser.add_argument(
        "--features-config",
        type=Path,
        default=Path("config/features.yml"),
        help="Path to features.yml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/model.yml"),
        help="Path to model.yml",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    try:
        stages = run_full_pipeline(
            preprocess_config=args.preprocess_config,
            feature_config=args.features_config,
            model_config=args.model_config,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        raise

    for name, path in stages:
        location = path if path is not None else ""
        print(f"{name}: {location}")


if __name__ == "__main__":
    main()
