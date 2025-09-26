# src/common/io.py
from pathlib import Path
import pandas as pd
from typing import List
from .app_logging import log, warn
import os


def save_parquet_or_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Save DataFrame to Parquet or CSV based on the file extension of `path`.
    Returns the final path used. Raises ValueError if extension unsupported.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".parquet":
        df.to_parquet(path, index=False)
        log("io.write_parquet", path=str(path), rows=int(df.shape[0]), cols=int(df.shape[1]))
        return path

    if ext == ".csv":
        df.to_csv(path, index=False)
        log("io.write_csv", path=str(path), rows=int(df.shape[0]), cols=int(df.shape[1]))
        return path

    raise ValueError(f"Unsupported file extension '{ext}'. Use .parquet or .csv.")


def load_parquet_or_csv(path: Path) -> pd.DataFrame:
    """
    Load DataFrame from Parquet or CSV based on the file extension of `path`.
    Raises FileNotFoundError if file does not exist, or ValueError if unsupported extension.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if ext == ".parquet":
        log("io.read_parquet", path=str(path))
        return pd.read_parquet(path)

    if ext == ".csv":
        log("io.read_csv", path=str(path))
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file extension '{ext}'. Use .parquet or .csv.")


def write_list(items: List[str], path) -> None:
    """Write a list of strings to a text file, one per line. Overwrites the file if it exists."""
    path = Path(path)
    with path.open("w") as f:
        for item in items:
            f.write(f"{item}\n")


def ensure_dir(path: str) -> str:
    """Ensure a directory exists; return the path."""
    os.makedirs(path, exist_ok=True)
    return path