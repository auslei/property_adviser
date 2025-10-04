"""Typed configuration for the derive stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from property_adviser.core.config import load_config


@dataclass(frozen=True)
class DeriveSettings:
    on_missing_source: str = "warn"
    output_conflict: str = "warn"
    default_fillna: Optional[Any] = None
    datasets: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepSpec:
    id: str
    type: str
    enabled: bool
    raw: Mapping[str, Any]

    @property
    def config(self) -> Mapping[str, Any]:
        return self.raw


@dataclass(frozen=True)
class DeriveConfig:
    settings: DeriveSettings
    steps: List[StepSpec]
    raw: Mapping[str, Any] = field(default_factory=dict)


def _normalise_step(step: Mapping[str, Any]) -> StepSpec:
    if not isinstance(step, Mapping):
        raise TypeError("Each derive step must be a mapping.")

    sid = str(step.get("id")) if step.get("id") is not None else None
    stype = str(step.get("type")) if step.get("type") is not None else None
    if not sid:
        raise ValueError("Derive step missing required 'id'.")
    if not stype:
        raise ValueError(f"Step '{sid}' missing required 'type'.")

    enabled = step.get("enabled", True)
    return StepSpec(id=sid, type=stype, enabled=bool(enabled), raw=step)


def derive_config_from_mapping(mapping: Mapping[str, Any]) -> DeriveConfig:
    if "spec_version" not in mapping:
        raise ValueError("New derive spec requires 'spec_version'.")

    version = int(mapping.get("spec_version"))
    if version != 1:
        raise ValueError(f"Unsupported derive spec version: {version}")

    settings_raw = mapping.get("settings") or {}
    datasets = settings_raw.get("datasets") or {}
    settings = DeriveSettings(
        on_missing_source=str(settings_raw.get("on_missing_source", "warn")),
        output_conflict=str(settings_raw.get("output_conflict", "warn")),
        default_fillna=settings_raw.get("default_fillna"),
        datasets=datasets if isinstance(datasets, Mapping) else {},
    )

    steps_raw = mapping.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError("Derive spec requires a non-empty 'steps' list.")

    steps = [_normalise_step(step) for step in steps_raw]
    return DeriveConfig(settings=settings, steps=steps, raw=mapping)


def load_derive_config(path: Path) -> DeriveConfig:
    data = load_config(path)
    return derive_config_from_mapping(data)


__all__ = [
    "DeriveSettings",
    "StepSpec",
    "DeriveConfig",
    "derive_config_from_mapping",
    "load_derive_config",
]
