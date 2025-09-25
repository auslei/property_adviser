# src/common/config.py
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    """
    Strictly load a YAML config. Raises if the file is missing, empty,
    or not a mapping. Provides clearer YAML parse errors.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        # Surface a concise parse error with file context
        raise ValueError(f"Failed to parse YAML at {path}: {e}") from e

    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(f"Config at {path} is empty or not a mapping")
    return cfg


def require(cfg: Dict[str, Any], *keys: str) -> Any:
    """
    Fetch a nested value (no defaults). Example:
        ds = require(cfg, "data_source")
        enc = require(cfg, "data_source", "encoding")
    Raises KeyError with a clear dotted path if any key is missing.
    """
    cur: Any = cfg
    path = []
    for k in keys:
        path.append(k)
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError("Missing required config key: " + ".".join(path))
        cur = cur[k]
    return cur