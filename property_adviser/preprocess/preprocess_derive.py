"""Backward-compatible wrappers around the derive engine."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import pandas as pd

from property_adviser.preprocess.derive.engine import DerivationResult, run_derivation
from property_adviser.preprocess.derive import legacy


def run_derivation_stage(
    cleaned: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    datasets: Optional[Mapping[str, Any]] = None,
) -> DerivationResult:
    """Execute the derive stage and return the enriched frame plus artefacts."""
    return run_derivation(cleaned, config, datasets=datasets)


def derive_features(cleaned: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Legacy helper returning only the derived dataframe."""
    result = run_derivation_stage(cleaned, config)
    return result.frame


def build_segments(
    derived: pd.DataFrame,
    config: Mapping[str, Any],
) -> Tuple[Optional[pd.DataFrame], Mapping[str, Any]]:
    """Legacy compatibility wrapper for segment generation."""
    if "spec_version" in config:
        # New engine handles segments internally via dedicated steps; return empty metadata for now.
        return None, {}
    return legacy.build_segments(derived, config)


extract_street = legacy.extract_street

__all__ = [
    "run_derivation_stage",
    "derive_features",
    "build_segments",
    "extract_street",
    "DerivationResult",
]
