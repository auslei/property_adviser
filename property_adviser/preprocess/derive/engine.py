"""Config-driven derive engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from property_adviser.core.app_logging import log
from property_adviser.preprocess.derive.config import DeriveConfig, derive_config_from_mapping
from property_adviser.preprocess.derive.steps import get_step_class
from property_adviser.preprocess.derive.steps.base import DeriveContext, StepResult
from property_adviser.preprocess.derive import legacy


@dataclass
class DerivationResult:
    frame: pd.DataFrame
    segments: Optional[pd.DataFrame] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


def _combine_datasets(config: DeriveConfig, overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    datasets: Dict[str, Any] = {}
    if config.settings.datasets:
        datasets.update(dict(config.settings.datasets))
    if overrides:
        datasets.update(dict(overrides))
    return datasets


def run_derivation(
    cleaned: pd.DataFrame,
    config_mapping: Mapping[str, Any],
    *,
    datasets: Optional[Mapping[str, Any]] = None,
) -> DerivationResult:
    """Run the derive pipeline against the cleaned dataframe."""
    if "spec_version" not in config_mapping:
        # Fallback to legacy behaviour
        derived = legacy.derive_features(cleaned.copy(), config_mapping)
        segments, _ = legacy.build_segments(derived, config_mapping)
        return DerivationResult(frame=derived, segments=segments, artifacts={})

    derive_config = derive_config_from_mapping(config_mapping)
    merged_datasets = _combine_datasets(derive_config, datasets)
    context = DeriveContext(settings=vars(derive_config.settings), datasets=merged_datasets)

    frame = cleaned.copy()
    artifacts: Dict[str, Any] = {}
    segments: Optional[pd.DataFrame] = None

    for step_spec in derive_config.steps:
        if not step_spec.enabled:
            continue
        step_cls = get_step_class(step_spec.type)
        step = step_cls(step_spec)
        result: StepResult = step.execute(frame, context)
        frame = result.frame
        if result.produced:
            artifacts[step_spec.id] = result.produced
        if result.segments is not None:
            segments = result.segments

    default_fill = derive_config.settings.default_fillna
    if default_fill is not None:
        frame = frame.fillna(default_fill)

    log("derive.engine.complete", steps=len(derive_config.steps), columns=frame.columns.tolist())
    return DerivationResult(frame=frame, segments=segments, artifacts=artifacts)


__all__ = ["run_derivation", "DerivationResult"]
