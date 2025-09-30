"""CLI entry point and convenience wrappers for preprocessing."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from property_adviser.core.app_logging import setup_logging, log
from property_adviser.config import PROJECT_ROOT
from property_adviser.preprocess import (
    PreprocessConfig,
    PreprocessResult,
    load_preprocess_config,
    run_preprocessing,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "preprocessing.yml"


def preprocess(cfg: Mapping[str, Any]) -> Path:
    """Backward-compatible wrapper around :func:`run_preprocessing`.

    Accepts the raw mapping loaded from ``preprocessing.yml`` and persists artefacts.
    Returns the path to the derived dataset.
    """
    config_dir = PROJECT_ROOT / "config"
    config = PreprocessConfig.from_mapping(
        cfg,
        base_path=PROJECT_ROOT,
        config_dir=config_dir if config_dir.exists() else PROJECT_ROOT,
    )
    result = run_preprocessing(config, write_outputs=True)
    return result.derived_path


def run_from_file(config_path: Path) -> PreprocessResult:
    config = load_preprocess_config(config_path)
    return run_preprocessing(config, write_outputs=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preprocessing (clean + derive) pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to preprocessing.yml",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    log("preprocess.cli.start", config=str(args.config), verbose=args.verbose)

    result = run_from_file(args.config)

    log(
        "preprocess.cli.complete",
        cleaned=str(result.cleaned_path),
        derived=str(result.derived_path),
        metadata=str(result.metadata_path),
        rows=int(result.derived.shape[0]),
        cols=int(result.derived.shape[1]),
    )
    print(f"Derived dataset written to {result.derived_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
