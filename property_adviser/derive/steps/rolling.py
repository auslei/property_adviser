"""Rolling operations along grouped time series."""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from property_adviser.core.app_logging import log, warn
from property_adviser.derive.steps.base import DeriveContext, DeriveStep, StepResult
from property_adviser.derive.steps.aggregate import AGG_DISPATCH


class RollingStep(DeriveStep):
    step_type = "rolling"

    def execute(self, frame: pd.DataFrame, context: DeriveContext) -> StepResult:
        cfg = dict(self.cfg)
        if not cfg.get("enabled", True):
            return StepResult(frame=frame)

        group_by = cfg.get("group_by") or cfg.get("groups")
        sort_by = cfg.get("sort_by")
        target = cfg.get("target")
        window = cfg.get("window")
        outputs = cfg.get("outputs")
        if not group_by or not sort_by or not target or not window or not outputs:
            raise ValueError(
                f"Rolling step '{self.spec.id}' requires 'group_by', 'sort_by', 'target', 'window', and 'outputs'."
            )

        group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        if sort_by not in frame.columns:
            warn("derive.rolling", step=self.spec.id, column=sort_by, reason="missing_sort_column")
            return StepResult(frame=frame)
        if target not in frame.columns:
            warn("derive.rolling", step=self.spec.id, column=target, reason="missing_target")
            return StepResult(frame=frame)

        config = cfg.get("config") if isinstance(cfg.get("config"), Mapping) else {}
        min_periods = config.get("min_periods", 1)
        center = bool(config.get("center", False))

        ordered = frame.sort_values(group_cols + [sort_by])
        grouped = ordered.groupby(group_cols, dropna=False)

        df = frame.copy()
        for agg_name, output in outputs.items():
            agg_key = str(agg_name).lower()
            if agg_key not in AGG_DISPATCH:
                raise ValueError(f"Unsupported rolling aggregate '{agg_name}' in step '{self.spec.id}'.")
            func = AGG_DISPATCH[agg_key]

            rolled = grouped[target].transform(
                lambda s: s.rolling(
                    window=int(window), min_periods=int(min_periods), center=center
                ).aggregate(func)
            )
            ordered_with_values = ordered.assign(__derive_roll=rolled)
            aligned = ordered_with_values.sort_index()["__derive_roll"].reindex(df.index)
            df[str(output)] = aligned

        df = df.copy()

        log("derive.rolling", step=self.spec.id, outputs=list(outputs.values()), window=window)
        return StepResult(frame=df)


__all__ = ["RollingStep"]
