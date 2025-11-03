
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from property_adviser.core.paths import PROJECT_ROOT

@dataclass
class GeocodeConfig:
    """Configuration for the geocoding process."""
    input_path: Path
    output_path: Path
    level: str

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> GeocodeConfig:
        """Create a geocode config from a dictionary."""
        return cls(
            input_path=PROJECT_ROOT / cfg["input_path"],
            output_path=PROJECT_ROOT / cfg["output_path"],
            level=cfg.get("level", "street"),
        )
