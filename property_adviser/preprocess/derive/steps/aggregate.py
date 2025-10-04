"""Groupby aggregation steps."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from property_adviser.core.app_logging import log, warn
from property_adviser.preprocess.derive.steps.base import DeriveContext, DeriveStep, StepResult


AGG_DISPATCH = {
    "mean": pd.Series.mean,
    "median": pd.Series.median,
    "sum": pd.Series.sum,
    "min": pd.Series.min,
    "max": pd.Series.max,
    "std": pd.Series.std,
    "count": pd.Series.count,
    "nunique": pd.Series.nunique,
}


class AggregateStep(DeriveStep):
    step_type = "aggregate"

    def execute(self, frame: pd.DataFrame, context: DeriveContext) -> StepResult:
        cfg = dict(self.cfg)
        if not cfg.get("enabled", True):
            return StepResult(frame=frame)

        group_by = cfg.get("group_by") or cfg.get("groups")
        if not group_by:
            raise ValueError(f"Aggregate step '{self.spec.id}' requires 'group_by'.")
        if isinstance(group_by, str):
            group_cols = [group_by]
        else:
            group_cols = list(group_by)

        target = cfg.get("target")
        if not target:
            raise ValueError(f"Aggregate step '{self.spec.id}' requires 'target'.")
        if target not in frame.columns:
            warn("derive.aggregate", step=self.spec.id, target=target, reason="missing_target")
            return StepResult(frame=frame)

        missing_groups = [col for col in group_cols if col not in frame.columns]
        if missing_groups:
            warn(
                "derive.aggregate",
                step=self.spec.id,
                reason="missing_group_columns",
                columns=missing_groups,
            )
            return StepResult(frame=frame)

        outputs_cfg = cfg.get("outputs")
        if not isinstance(outputs_cfg, Mapping) or not outputs_cfg:
            raise ValueError(f"Aggregate step '{self.spec.id}' requires 'outputs' mapping.")

        config = cfg.get("config") if isinstance(cfg.get("config"), Mapping) else {}
        min_count = config.get("min_count")
        grouped = frame.groupby(group_cols, dropna=False)
        df = frame.copy()

        for agg_name, output in outputs_cfg.items():
            agg_name_lower = str(agg_name).lower()
            if agg_name_lower not in AGG_DISPATCH:
                raise ValueError(f"Unsupported aggregate '{agg_name}' in step '{self.spec.id}'.")
            func = AGG_DISPATCH[agg_name_lower]
            agg_series = grouped[target].transform(func)
            if min_count is not None:
                counts = grouped[target].transform("count")
                agg_series = agg_series.where(counts >= int(min_count))
            df[str(output)] = agg_series

        log("derive.aggregate", step=self.spec.id, groups=len(group_cols), outputs=list(outputs_cfg.values()))
        return StepResult(frame=df)


__all__ = ["AggregateStep"]
