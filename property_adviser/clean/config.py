
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from property_adviser.core.paths import PROJECT_ROOT

@dataclass
class CleanConfig:
    """Configuration for the cleaning process."""
    input_path: Path
    output_path: Path
    dropped_rows_path: Path
    config: Dict[str, Any]

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> CleanConfig:
        """Create a clean config from a dictionary."""
        return cls(
            input_path=PROJECT_ROOT / cfg["data_source"]["path"],
            output_path=PROJECT_ROOT / cfg["output_path"],
            dropped_rows_path=PROJECT_ROOT / cfg["dropped_rows_path"],
            config=cfg,
        )
