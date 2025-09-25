from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Loads a YAML file and returns its content as a dictionary.

    Args:
        path: The path to the YAML file.

    Returns:
        A dictionary with the content of the YAML file.

    Raises:
        ValueError: If the file does not exist, is not a valid YAML file, or is not a mapping.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            content = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration at {path}: {exc}") from exc
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(f"Configuration at {path} must be a mapping.")
    return content