"""Configuration schema helpers for the preprocessing stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from property_adviser.core.config import load_config


@dataclass(frozen=True)
class CleanStageConfig:
    config_path: Path
    output_path: Path
    dropped_rows_path: Optional[Path] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "output_path": str(self.output_path),
            "dropped_rows_path": str(self.dropped_rows_path) if self.dropped_rows_path else None,
        }


@dataclass(frozen=True)
class DeriveStageConfig:
    config_path: Path
    output_path: Path
    metadata_path: Path
    segment_output_path: Optional[Path] = None
    detailed_output_path: Optional[Path] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "output_path": str(self.output_path),
            "metadata_path": str(self.metadata_path),
            "segment_output_path": str(self.segment_output_path) if self.segment_output_path else None,
            "detailed_output_path": str(self.detailed_output_path) if self.detailed_output_path else None,
        }


@dataclass(frozen=True)
class PreprocessConfig:
    clean: CleanStageConfig
    derive: DeriveStageConfig
    options: Mapping[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean.to_dict(),
            "derivation": self.derive.to_dict(),
            "options": dict(self.options),
            "source_path": str(self.source_path) if self.source_path else None,
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_path: Optional[Path] = None,
        config_dir: Optional[Path] = None,
        source_path: Optional[Path] = None,
    ) -> "PreprocessConfig":
        base = base_path or Path.cwd()
        cfg_dir = config_dir or base
        clean_cfg = mapping.get("clean") or {}
        derivation_cfg = mapping.get("derivation") or mapping.get("derive") or {}

        if "config_path" not in clean_cfg or "output_path" not in clean_cfg:
            raise KeyError("Cleaning stage requires 'config_path' and 'output_path'.")
        if "config_path" not in derivation_cfg or "output_path" not in derivation_cfg or "metadata_path" not in derivation_cfg:
            raise KeyError("Derivation stage requires 'config_path', 'output_path', and 'metadata_path'.")

        def resolve(path_value: Any) -> Path:
            path = Path(path_value)
            if path.is_absolute():
                return path
            parts = path.parts
            if cfg_dir and parts and parts[0] == cfg_dir.name:
                anchor = cfg_dir.parent
                return anchor / path if anchor else cfg_dir / path
            return base / path

        clean_stage = CleanStageConfig(
            config_path=resolve(clean_cfg["config_path"]),
            output_path=resolve(clean_cfg["output_path"]),
            dropped_rows_path=resolve(clean_cfg["dropped_rows_path"]) if clean_cfg.get("dropped_rows_path") else None,
        )

        derive_stage = DeriveStageConfig(
            config_path=resolve(derivation_cfg["config_path"]),
            output_path=resolve(derivation_cfg["output_path"]),
            metadata_path=resolve(derivation_cfg["metadata_path"]),
            segment_output_path=resolve(derivation_cfg["segment_output_path"]) if derivation_cfg.get("segment_output_path") else None,
            detailed_output_path=resolve(derivation_cfg["detailed_output_path"]) if derivation_cfg.get("detailed_output_path") else None,
        )

        options = mapping.get("options") or {}
        return cls(clean=clean_stage, derive=derive_stage, options=options, source_path=source_path)


def load_preprocess_config(config_path: Path) -> PreprocessConfig:
    """Load preprocessing.yml into a typed configuration object."""
    raw = load_config(config_path)
    config_dir = config_path.parent
    project_root = config_dir.parent if config_dir.name == "config" else config_dir
    return PreprocessConfig.from_mapping(
        raw,
        base_path=project_root,
        config_dir=config_dir,
        source_path=config_path,
    )


__all__ = [
    "CleanStageConfig",
    "DeriveStageConfig",
    "PreprocessConfig",
    "load_preprocess_config",
]
