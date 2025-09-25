# preprocess.py
import argparse, json
from pathlib import Path
import pandas as pd
from src.common.app_logging import *
from src.common.config import load_config
from src.common.io import save_parquet_or_csv
from src.preprocess_clean import clean_data
from src.preprocess_derive import derive_features


DATA_DIR = Path("data")
PREPROCESS_DIR = Path("data_preprocess"); PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = Path("config")
PREPROCESS_CONFIG_PATH = CONFIG_DIR / "preprocessing.yml"
DERIVED_OUTPUT_PATH = PREPROCESS_DIR / "derived.parquet"
CLEANED_OUTPUT_PATH = PREPROCESS_DIR / "cleaned.parquet"
METADATA_PATH = PREPROCESS_DIR / "metadata.json"

def _make_metadata(df: pd.DataFrame, cfg: dict) -> dict:
    num = df.select_dtypes(include=["number"]).columns.tolist()
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    sources = sorted(df["__source_file"].dropna().unique().tolist()) if "__source_file" in df.columns else []
    return {"rows": int(df.shape[0]), "columns": df.columns.tolist(),
            "numeric_columns": num, "categorical_columns": cat,
            "sources": sources, "configuration": cfg}

def preprocess(cfg: dict) -> Path:
    cleaned = clean_data(cfg)
    cleaned_out = save_parquet_or_csv(cleaned, CLEANED_OUTPUT_PATH)
    log("dataset.clean_saved", path=str(cleaned_out), columns=list(cleaned.columns))

    derived = derive_features(cleaned.copy(), cfg)
    log("dataset.final_columns", columns=list(derived.columns))
    if "priceFactor" in derived.columns:
        log("stats.price_factor", min=float(derived["priceFactor"].min()),
            max=float(derived["priceFactor"].max()), mean=float(derived["priceFactor"].mean()))

    derived_out = save_parquet_or_csv(derived, DERIVED_OUTPUT_PATH)
    metadata = _make_metadata(derived, cfg); METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    log("io.write_metadata", path=str(METADATA_PATH))
    return derived_out

def main():
    parser = argparse.ArgumentParser(description="Standalone data preprocess (clean + derive).")
    parser.add_argument("--config", type=str, default=str(PREPROCESS_CONFIG_PATH))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    cfg = load_config(Path(args.config))
    out = preprocess(cfg)
    print(f"Preprocessed data saved to {out}")

if __name__ == "__main__":
    main()