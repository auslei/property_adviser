
"""CLI entry point for the derivation stage."""
from __future__ import annotations

import argparse
from pathlib import Path

from property_adviser.core.app_logging import setup_logging, log
from property_adviser.derive.config import load_derive_config
from property_adviser.derive.engine import run_derivation

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "derive.yml"

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the derivation pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to derive.yml",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    log("derive.cli.start", config=str(args.config), verbose=args.verbose)

    config = load_derive_config(args.config)
    run_derivation(config)

    log("derive.cli.complete")

if __name__ == "__main__":
    main()
