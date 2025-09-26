from __future__ import annotations

import argparse
from typing import Any, Dict, Optional
from pathlib import Path

from property_adviser.train.model_training import train_timeseries_model
from property_adviser.core.config import load_config
from property_adviser.core.app_logging import log, setup_logging

CONFIG_PATH = "config/model.yml"

def train_models(config: Optional[Dict[str, Any]] = None):
    """
    Train models for Property Adviser.
    - Reads X, y, optional feature_scores from model.yml
    - Grid-searches candidates
    - Saves timestamped best model + model scores CSV
    - Returns a dict suitable for GUI visualisation
    """
    config_path = None
    overrides = config or {}
    return train_timeseries_model(config_path, overrides)

def main():
    parser = argparse.ArgumentParser(description="Standalone model training")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to features.yml config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    
    outcome = train_timeseries_model(config_path=args.config, overrides={"verbose": args.verbose})
    # Concise console line; full details in returned outcome for GUI
    print(f"Best: {outcome['best_model']}  "
          f"Model: {outcome['best_model_path']}  "
          f"Scores: {outcome['scores_path']}  "
          f"ValMonth: {outcome['validation_month']}")

    
    train_models(cfg)


if __name__ == "__main__":
    main()