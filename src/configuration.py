from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
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


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON at {path}: {exc}") from exc
