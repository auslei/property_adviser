
"""CLI entry point for geocoding."""
from __future__ import annotations

import argparse
from pathlib import Path

from property_adviser.core.app_logging import setup_logging, log
from property_adviser.core.config import load_geocode_config
from property_adviser.geocode.main import run_geocoding

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "geocode.yml"

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the geocoding pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to geocode.yml",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    log("geocode.cli.start", config=str(args.config), verbose=args.verbose)

    config = load_geocode_config(args.config)
    run_geocoding(config)

    log("geocode.cli.complete")

if __name__ == "__main__":
    main()
