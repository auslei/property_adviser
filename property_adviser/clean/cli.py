
"""CLI entry point for the cleaning stage."""
from __future__ import annotations

import argparse
from pathlib import Path

from property_adviser.core.app_logging import setup_logging, log
from property_adviser.clean.config import CleanConfig
from property_adviser.clean.engine import clean_data
from property_adviser.core.config import load_config

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "clean.yml"

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cleaning pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to clean.yml",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    log("clean.cli.start", config=str(args.config), verbose=args.verbose)

    config_dict = load_config(args.config)
    config = CleanConfig.from_dict(config_dict)
    clean_data(config)

    log("clean.cli.complete")

if __name__ == "__main__":
    main()
