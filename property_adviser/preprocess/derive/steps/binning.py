"""Binning / categorisation step."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from property_adviser.core.app_logging import log, warn
from property_adviser.preprocess.derive.steps.base import DeriveContext, DeriveStep, StepResult


class BinStep(DeriveStep):
    step_type = "bin"

    def execute(self, frame: pd.DataFrame, context: DeriveContext) -> StepResult:
        cfg = dict(self.cfg)
        if not cfg.get("enabled", True):
            return StepResult(frame=frame)

        source = cfg.get("source")
        output = cfg.get("output")
        method = cfg.get("method", "fixed")
        config = cfg.get("config") if isinstance(cfg.get("config"), Mapping) else {}
        if not source or not output:
            raise ValueError(f"Bin step '{self.spec.id}' requires 'source' and 'output'.")
        if source not in frame.columns:
            warn("derive.bin", step=self.spec.id, source=source, reason="missing_source")
            return StepResult(frame=frame)

        df = frame.copy()
        series = df[source]

        if method == "fixed":
            edges = config.get("edges") or config.get("bins")
            labels = config.get("labels")
            fill_value = config.get("fill_value", "Unknown")
            include_lowest = bool(config.get("include_lowest", True))
            right = bool(config.get("right", True))
            if not edges:
                raise ValueError(f"Bin step '{self.spec.id}' missing 'edges'.")
            values = pd.to_numeric(series, errors="coerce")
            edges = sorted(float(edge) for edge in edges)
            bins = [-np.inf] + edges + [np.inf]
            categories = pd.cut(
                values,
                bins=bins,
                labels=labels if labels and len(labels) == len(bins) - 1 else None,
                include_lowest=include_lowest,
                right=right,
            )
            df[output] = categories.astype(str).replace({"nan": np.nan}).fillna(fill_value)
        elif method == "mapping":
            mapping = config.get("mapping") or {}
            default = config.get("default", "Unknown")
            df[output] = series.map(mapping).fillna(default)
        else:
            raise ValueError(f"Unsupported bin method '{method}' in step '{self.spec.id}'.")

        log("derive.bin", step=self.spec.id, output=output)
        return StepResult(frame=df)


__all__ = ["BinStep"]
