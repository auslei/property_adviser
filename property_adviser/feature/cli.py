"""CLI entry point for feature selection."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from property_adviser.config import PROJECT_ROOT
from property_adviser.core.app_logging import log, setup_logging
from property_adviser.feature.config import FeatureSelectionConfig, load_feature_selection_config
from property_adviser.feature.pipeline import FeatureSelectionResult, run_feature_selection as _run_with_config

DEFAULT_CONFIG_PATH = Path("config/features.yml")


def _ensure_config(cfg: FeatureSelectionConfig | Mapping[str, Any]) -> FeatureSelectionConfig:
    if isinstance(cfg, FeatureSelectionConfig):
        return cfg
    config_dir = PROJECT_ROOT / "config"
    base_path = PROJECT_ROOT
    return FeatureSelectionConfig.from_mapping(
        cfg,
        base_path=base_path,
        config_dir=config_dir if config_dir.exists() else base_path,
    )


def run_feature_selection(
    cfg: FeatureSelectionConfig | Mapping[str, Any],
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    use_top_k: Optional[bool] = None,
    top_k: Optional[int] = None,
    write_outputs: bool = True,
    scores_output_filename: Optional[str] = None,
) -> FeatureSelectionResult:
    config = _ensure_config(cfg)
    return _run_with_config(
        config,
        include=include,
        exclude=exclude,
        use_top_k=use_top_k,
        top_k=top_k,
        write_outputs=write_outputs,
        scores_output_filename=scores_output_filename,
    )


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Standalone feature selection.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to features.yml")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--scores-file",
        type=str,
        default="feature_scores.parquet",
        help="Output filename for the full scores table (parquet or csv).",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    configs = load_feature_selection_config(args.config)
    log(
        "feature_selection.cli.start",
        config=str(args.config),
        verbose=args.verbose,
        targets=len(configs),
    )

    summaries = []
    for cfg in configs:
        result = _run_with_config(
            cfg,
            include=None,
            exclude=None,
            use_top_k=None,
            top_k=None,
            write_outputs=True,
            scores_output_filename=args.scores_file if len(configs) == 1 else None,
        )
        print(
            f"Target {cfg.target}: selected {len(result.selected_columns)} features → {result.output_dir}"
        )
        if result.scores_path:
            print(f"  Scores table → {result.scores_path}")
        summaries.append(
            {
                "target": cfg.target,
                "selected_columns": result.selected_columns,
                "n_selected": len(result.selected_columns),
                "output_dir": str(result.output_dir) if result.output_dir else None,
                "scores_path": str(result.scores_path) if result.scores_path else None,
            }
        )

    return {"targets": summaries}


if __name__ == "__main__":  # pragma: no cover
    main()
