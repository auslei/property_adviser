"""Base abstractions for derive steps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from property_adviser.preprocess.derive.config import StepSpec


@dataclass
class StepResult:
    frame: pd.DataFrame
    produced: Mapping[str, Any] = field(default_factory=dict)
    segments: Optional[pd.DataFrame] = None


class DeriveStep:
    """Protocol-like base class for derive steps."""

    step_type: str = ""

    def __init__(self, spec: StepSpec) -> None:
        self.spec = spec

    @property
    def cfg(self) -> Mapping[str, Any]:
        return self.spec.config

    def execute(self, frame: pd.DataFrame, context: "DeriveContext") -> StepResult:
        raise NotImplementedError


@dataclass
class DeriveContext:
    settings: Mapping[str, Any]
    datasets: Dict[str, Any] = field(default_factory=dict)

    def load_dataset(self, key: Any) -> pd.DataFrame:
        from property_adviser.core.io import load_parquet_or_csv

        if isinstance(key, pd.DataFrame):
            return key
        if isinstance(key, str):
            # Literal path provided
            if any(sep in key for sep in ("/", "\\")) or key.lower().endswith((".csv", ".parquet")):
                return load_parquet_or_csv(key)
            if key in self.datasets:
                value = self.datasets[key]
                if isinstance(value, pd.DataFrame):
                    return value
                return load_parquet_or_csv(value)
        if isinstance(key, Mapping):
            source = key.get("path")
            if source:
                return load_parquet_or_csv(source)
            alias = key.get("dataset")
            if alias and alias in self.datasets:
                value = self.datasets[alias]
                if isinstance(value, pd.DataFrame):
                    return value
                return load_parquet_or_csv(value)
        raise KeyError(f"Unable to resolve dataset '{key}' for join step.")


__all__ = ["DeriveStep", "StepResult", "DeriveContext"]
