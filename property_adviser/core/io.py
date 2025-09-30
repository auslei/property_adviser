# src/common/io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd

from .app_logging import log


def save_parquet_or_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Save DataFrame to Parquet or CSV based on the file extension of `path`.
    Returns the final path used. Raises ValueError if extension unsupported.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_list(items: Iterable[str], path: Path | str) -> None:
    """Write a list of strings to a text file, one per line. Overwrites the file if it exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")

def ensure_dir(path: Path | str) -> Path:
    """Ensure a directory exists; return the resolved Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def read_json(path: Path | str) -> Any:
    """Load JSON content from a file, raising if the file does not exist."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path_obj}")
    with path_obj.open("r", encoding="utf-8") as handle:
        return json.load(handle)
