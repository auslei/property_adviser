# src/common/io.py
from pathlib import Path
import pandas as pd
from .app_logging import log, warn


def save_parquet_or_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Save DataFrame to Parquet at `path`. If Parquet write fails, fall back to CSV
    with the same name but .csv extension. Returns the final path used.
    """
    try:
        df.to_parquet(path, index=False)
        log("io.write_parquet", path=str(path), rows=int(df.shape[0]), cols=int(df.shape[1]))
        return path
    except Exception as e:
        alt = path.with_suffix(".csv")
        warn("io.write_parquet", path=str(path), error=str(e), fallback=str(alt))
        df.to_csv(alt, index=False)
        log("io.write_csv", path=str(alt), rows=int(df.shape[0]), cols=int(df.shape[1]))
        return alt


def load_parquet_or_csv(path: Path) -> pd.DataFrame:
    """
    Load DataFrame from Parquet if available, otherwise fall back to CSV.
    Raises FileNotFoundError if neither exists.
    """
    parquet_path = path.with_suffix(".parquet")
    csv_path = path.with_suffix(".csv")

    if parquet_path.exists():
        log("io.read_parquet", path=str(parquet_path))
        return pd.read_parquet(parquet_path)

    if csv_path.exists():
        log("io.read_csv", path=str(csv_path))
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} exists.")