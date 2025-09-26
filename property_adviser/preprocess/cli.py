# cli.py (preprocess stage)
# ---------------------------------------------------
# This file provides a command-line entry point for the
# "preprocess" stage of the pipeline. It ties together:
#   1. Cleaning raw data (preprocess_clean)
#   2. Deriving additional features (preprocess_derive)
#   3. Saving final outputs + metadata
# ---------------------------------------------------

import argparse
import json
from pathlib import Path
import pandas as pd

from property_adviser.core.app_logging import setup_logging, log
from property_adviser.core.config import load_config
from property_adviser.core.io import save_parquet_or_csv
from property_adviser.preprocess.preprocess_clean import clean_data
from property_adviser.preprocess.preprocess_derive import derive_features

# Default config location (relative to repo root)
PREPROCESS_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "preprocessing.yml"


# ---------------------------------------------------
# Metadata helper
# ---------------------------------------------------
def _make_metadata(df: pd.DataFrame, cfg: dict) -> dict:
    """Build metadata dictionary summarising the dataset."""
    num = df.select_dtypes(include=["number"]).columns.tolist()
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    sources = (
        sorted(df["__source_file"].dropna().unique().tolist())
        if "__source_file" in df.columns else []
    )

    return {
        "rows": int(df.shape[0]),
        "columns": df.columns.tolist(),
        "numeric_columns": num,
        "categorical_columns": cat,
        "sources": sources,
        "configuration": cfg,
    }


# ---------------------------------------------------
# Main pipeline function
# ---------------------------------------------------
def preprocess(cfg: dict) -> Path:
    """
    Run preprocessing pipeline:
      - Clean raw data
      - Derive new features
      - Save outputs + metadata
    Returns the path to the final derived dataset.
    """
    # --- CLEAN ---
    log("preprocess.start, stage='Loading cleaning config', path=cfg['clean']['config_path']")
    cleaning_cfg = load_config(Path(cfg["clean"]["config_path"]))
    
    cleaned = clean_data(cleaning_cfg)
    cleaned_out = save_parquet_or_csv(cleaned, cfg["clean"]["output_path"])
    log("dataset.clean_saved", path=str(cleaned_out), columns=list(cleaned.columns))

    # --- DERIVE ---
    derivation_cfg = load_config(Path(cfg["derivation"]["config_path"]))
    derived = derive_features(cleaned.copy(), derivation_cfg)
    log("dataset.final_columns", columns=list(derived.columns))

    # Log some example stats
    if "priceFactor" in derived.columns:
        log(
            "stats.price_factor",
            min=float(derived["priceFactor"].min()),
            max=float(derived["priceFactor"].max()),
            mean=float(derived["priceFactor"].mean()),
        )

    # --- METADATA ---
    # If cfg["metadata_path"] exists, use it; otherwise default to same path as derived with .json suffix
    metadata_path = Path(cfg["derivation"]["metadata_path"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = _make_metadata(derived, cfg)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    log("io.write_metadata", path=str(metadata_path))

    # --- SAVE DERIVED DATASET ---
    derived_out = save_parquet_or_csv(derived, cfg["derivation"]["output_path"])
    log("dataset.derived_saved", path=str(derived_out), columns=list(derived.columns))

    return derived_out


# ---------------------------------------------------
# CLI entry point
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Standalone data preprocess (clean + derive).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PREPROCESS_CONFIG_PATH),
        help="Path to preprocessing.yml config file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Setup logging first
    setup_logging(verbose=args.verbose)

    # Load config YAML
    cfg = load_config(Path(args.config))
    log("io.config_loaded", path=args.config)

    # Run pipeline
    out = preprocess(cfg)
    print(f"Preprocessed data saved to {out}")


if __name__ == "__main__":
    main()
