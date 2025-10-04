"""Dataset join step."""
from __future__ import annotations

from typing import List, Mapping

import pandas as pd

from property_adviser.core.app_logging import log, warn
from property_adviser.preprocess.derive.steps.base import DeriveContext, DeriveStep, StepResult


class JoinStep(DeriveStep):
    step_type = "join"

    def execute(self, frame: pd.DataFrame, context: DeriveContext) -> StepResult:
        cfg = dict(self.cfg)
        if not cfg.get("enabled", True):
            return StepResult(frame=frame)

        right_cfg = cfg.get("right")
        if right_cfg is None:
            raise ValueError(f"Join step '{self.spec.id}' requires 'right'.")

        right_df = context.load_dataset(right_cfg)
        if not isinstance(right_df, pd.DataFrame):
            raise ValueError(f"Join step '{self.spec.id}' failed to load dataset '{right_cfg}'.")

        on_cfg = cfg.get("on")
        if not on_cfg:
            raise ValueError(f"Join step '{self.spec.id}' requires 'on'.")

        left_on: List[str] = []
        right_on: List[str] = []
        for item in on_cfg:
            if isinstance(item, Mapping):
                left_col = item.get("left") or item.get("column")
                right_col = item.get("right") or left_col
                if not left_col or not right_col:
                    raise ValueError(f"Join step '{self.spec.id}' has invalid join mapping {item}.")
            else:
                left_col = str(item)
                right_col = left_col
            left_on.append(left_col)
            right_on.append(right_col)

        missing = [col for col in left_on if col not in frame.columns]
        if missing:
            warn("derive.join", step=self.spec.id, reason="missing_left_columns", columns=missing)
            return StepResult(frame=frame)

        right_missing = [col for col in right_on if col not in right_df.columns]
        if right_missing:
            warn("derive.join", step=self.spec.id, reason="missing_right_columns", columns=right_missing)
            return StepResult(frame=frame)

        select_cols = cfg.get("select")
        if select_cols:
            keep_cols = set(select_cols) | set(right_on)
            right_df = right_df.loc[:, [c for c in right_df.columns if c in keep_cols]]

        how = cfg.get("how", "left")
        suffixes = cfg.get("suffixes", ("", "_right"))
        validate = cfg.get("validate")

        merged = frame.merge(
            right_df,
            how=how,
            left_on=left_on,
            right_on=right_on,
            suffixes=tuple(suffixes) if isinstance(suffixes, (list, tuple)) else ("", "_right"),
            validate=validate,
        )

        log(
            "derive.join",
            step=self.spec.id,
            how=how,
            left_on=left_on,
            right_on=right_on,
            columns_added=[c for c in merged.columns if c not in frame.columns],
        )
        return StepResult(frame=merged)


__all__ = ["JoinStep"]
