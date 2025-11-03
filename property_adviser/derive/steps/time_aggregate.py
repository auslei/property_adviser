"""Time-windowed aggregations for grouped series."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd

from property_adviser.core.app_logging import log, warn
from property_adviser.derive.steps.base import DeriveContext, DeriveStep, StepResult


AGG_FUNCS: Dict[str, Callable[[pd.Series], float]] = {
    "mean": pd.Series.mean,
    "median": pd.Series.median,
    "sum": pd.Series.sum,
    "min": pd.Series.min,
    "max": pd.Series.max,
    "std": pd.Series.std,
    "count": pd.Series.count,
}


def _normalise_month_value(value: float | int | str) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        if value > 1000:  # assume YYYYMM
            year = value // 100
            month = value % 100
            return year * 12 + (month - 1)
        return float(value)
    text = str(value)
    if not text:
        return np.nan
    text = text.strip()
    if text.isdigit() and len(text) >= 6:
        year = int(text[:4])
        month = int(text[4:6])
        return year * 12 + (month - 1)
    try:
        dt = pd.Period(text, freq="M")
        return float(dt.year * 12 + (dt.month - 1))
    except Exception:
        try:
            parsed = pd.to_datetime(text)
            return float(parsed.year * 12 + (parsed.month - 1))
        except Exception:
            return np.nan


def _normalise_time(series: pd.Series, unit: str) -> pd.Series:
    if unit.lower() != "months":
        raise ValueError(f"Unsupported time unit '{unit}'. Only 'months' supported.")
    return series.apply(_normalise_month_value)


def _window_bounds(past: int, future: int, include_current: bool) -> Tuple[int, int, bool]:
    past = max(0, int(past))
    future = max(0, int(future))
    return past, future, bool(include_current)


def _apply_time_window(
    group: pd.DataFrame,
    *,
    time_col: str,
    target: str,
    aggregator: Callable[[pd.Series], float],
    past: int,
    future: int,
    include_current: bool,
) -> pd.Series:
    times = group[time_col].to_numpy()
    values = pd.to_numeric(group[target], errors="coerce").to_numpy()
    result = np.full(len(group), np.nan)

    for idx in range(len(group)):
        current_time = times[idx]
        if np.isnan(current_time):
            continue
        lower = current_time - past
        upper = current_time + future
        mask = (times >= lower) & (times <= upper)
        if not include_current:
            mask[idx] = False
        window_values = values[mask]
        if window_values.size == 0:
            continue
        series = pd.Series(window_values)
        agg_value = aggregator(series)
        result[idx] = agg_value

    return pd.Series(result, index=group.index)


class TimeAggregateStep(DeriveStep):
    step_type = "time_aggregate"

    def execute(self, frame: pd.DataFrame, context: DeriveContext) -> StepResult:
        cfg = dict(self.cfg)
        if not cfg.get("enabled", True):
            return StepResult(frame=frame)

        group_by = cfg.get("group_by") or cfg.get("groups")
        if not group_by:
            raise ValueError(f"Time aggregate step '{self.spec.id}' requires 'group_by'.")
        group_cols = [group_by] if isinstance(group_by, str) else list(group_by)

        time_col = cfg.get("time_col")
        target = cfg.get("target")
        if not time_col or not target:
            raise ValueError(f"Time aggregate step '{self.spec.id}' requires 'time_col' and 'target'.")
        if time_col not in frame.columns:
            warn("derive.time_aggregate", step=self.spec.id, column=time_col, reason="missing_time_col")
            return StepResult(frame=frame)
        if target not in frame.columns:
            warn("derive.time_aggregate", step=self.spec.id, column=target, reason="missing_target")
            return StepResult(frame=frame)

        outputs = cfg.get("outputs")
        if not isinstance(outputs, Mapping) or not outputs:
            raise ValueError(f"Time aggregate step '{self.spec.id}' requires 'outputs'.")

        window_cfg = cfg.get("window") or {}
        unit = str(window_cfg.get("unit", "months"))
        past, future, include_current = _window_bounds(
            window_cfg.get("past", 0),
            window_cfg.get("future", 0),
            window_cfg.get("include_current", False),
        )

        normalised = frame.copy()
        normalised["__derive_time"] = _normalise_time(normalised[time_col], unit)
        normalised = normalised.sort_values(group_cols + ["__derive_time"])

        df = frame.copy()

        for agg_name, output in outputs.items():
            agg_name_lower = str(agg_name).lower()
            if agg_name_lower not in AGG_FUNCS:
                raise ValueError(
                    f"Unsupported aggregation '{agg_name}' in time step '{self.spec.id}'."
                )
            func = AGG_FUNCS[agg_name_lower]

            result = pd.Series(np.nan, index=df.index)
            for _, group in normalised.groupby(group_cols, sort=False, dropna=False):
                window_series = _apply_time_window(
                    group,
                    time_col="__derive_time",
                    target=target,
                    aggregator=func,
                    past=past,
                    future=future,
                    include_current=include_current,
                )
                result.loc[group.index] = window_series.values

            df[str(output)] = result

        df = df.drop(columns="__derive_time", errors="ignore")
        log(
            "derive.time_aggregate",
            step=self.spec.id,
            outputs=list(outputs.values()),
            past=past,
            future=future,
            include_current=include_current,
        )
        return StepResult(frame=df)


__all__ = ["TimeAggregateStep"]
