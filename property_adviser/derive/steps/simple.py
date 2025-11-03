"""Simple transformation steps: expressions, mappings, utility transforms."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from property_adviser.core.app_logging import log, warn
from property_adviser.derive.steps.base import DeriveContext, DeriveStep, StepResult
from property_adviser.derive import legacy


def _as_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, np.ndarray):
        if value.shape[0] != len(index):
            raise ValueError("Expression helper received array with mismatched length.")
        return pd.Series(value, index=index)
    return pd.Series([value] * len(index), index=index)


def _nullif(a: Any, b: Any, *, index: pd.Index) -> pd.Series:
    lhs = _as_series(a, index)
    rhs = _as_series(b, index)
    return lhs.mask(lhs.eq(rhs))


def _clip(value: Any, lower: Any, upper: Any, *, index: pd.Index) -> pd.Series:
    series = _as_series(value, index)
    return series.clip(lower=lower, upper=upper)


def _max_series(*args: Any, index: pd.Index) -> pd.Series:
    if not args:
        raise ValueError("max requires at least one argument")
    stacked = pd.concat([_as_series(arg, index) for arg in args], axis=1)
    return stacked.max(axis=1)


def _min_series(*args: Any, index: pd.Index) -> pd.Series:
    if not args:
        raise ValueError("min requires at least one argument")
    stacked = pd.concat([_as_series(arg, index) for arg in args], axis=1)
    return stacked.min(axis=1)


SAFE_FUNCTIONS = {
    "nullif": _nullif,
    "clip": _clip,
    "max": _max_series,
    "min": _min_series,
    "maximum": _max_series,
    "minimum": _min_series,
    "abs": np.abs,
    "log": np.log,
    "log1p": np.log1p,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "where": np.where,
}


def _distance_to_cbd(lat1, lon1, lat2=-37.814, lon2=144.963):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


class SimpleStep(DeriveStep):
    step_type = "simple"

    def execute(self, frame: pd.DataFrame, context: DeriveContext) -> StepResult:
        cfg = dict(self.cfg)
        if not cfg.get("enabled", True):
            return StepResult(frame=frame)

        method = (cfg.get("method") or "").lower()
        if not method:
            raise ValueError(f"Simple step '{self.spec.id}' missing 'method'.")

        if method == "expr":
            return StepResult(frame=self._run_expr(frame, cfg))
        if method == "map_values":
            return StepResult(frame=self._run_map_values(frame, cfg))
        if method == "extract_street":
            return StepResult(frame=self._run_extract_street(frame, cfg))
        if method == "to_month_index":
            return StepResult(frame=self._run_month_index(frame, cfg))
        if method == "cyclical_encode":
            return StepResult(frame=self._run_month_cyclical(frame, cfg))
        if method == "property_age":
            return StepResult(frame=self._run_property_age(frame, cfg))
        if method == "date_parts":
            return StepResult(frame=self._run_date_parts(frame, cfg))
        if method == "distance_to_cbd":
            return StepResult(frame=self._run_distance_to_cbd(frame, cfg))

        raise ValueError(f"Unsupported simple method '{method}' in step '{self.spec.id}'.")

    # ------------------------------------------------------------------
    def _run_expr(self, frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
        expr = cfg.get("config", {}).get("expr") if isinstance(cfg.get("config"), Mapping) else cfg.get("expr")
        if not expr:
            raise ValueError(f"Expression step '{self.spec.id}' missing 'expr'.")
        output = cfg.get("output")
        if not output:
            raise ValueError(f"Expression step '{self.spec.id}' missing 'output'.")

        local_env: Dict[str, Any] = {col: frame[col] for col in frame.columns}
        index = frame.index

        def wrap_func(func):
            def inner(*args):
                return func(*args, index=index)
            return inner

        needs_index = {"nullif", "clip", "max", "min", "maximum", "minimum"}
        safe_env: Dict[str, Any] = {
            name: wrap_func(func) if name in needs_index else func
            for name, func in SAFE_FUNCTIONS.items()
        }
        safe_env["np"] = np
        safe_env.update(local_env)

        try:
            result = eval(expr, {"__builtins__": {}}, safe_env)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            raise ValueError(f"Failed to evaluate expression for step '{self.spec.id}': {exc}") from exc

        if isinstance(output, (list, tuple)):
            if not isinstance(result, (list, tuple)) or len(result) != len(output):
                raise ValueError(
                    f"Expression step '{self.spec.id}' expected {len(output)} outputs, got {result}."
                )
            frame = frame.copy()
            for col_name, values in zip(output, result):
                frame[col_name] = _as_series(values, frame.index)
        else:
            frame = frame.copy()
            frame[str(output)] = _as_series(result, frame.index)
        return frame

    def _run_map_values(self, frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
        source = cfg.get("source")
        output = cfg.get("output", source)
        mapping = cfg.get("config", {}).get("mapping") if isinstance(cfg.get("config"), Mapping) else cfg.get("mapping")
        default = cfg.get("config", {}).get("default") if isinstance(cfg.get("config"), Mapping) else cfg.get("default")
        if source is None or output is None:
            raise ValueError(f"Mapping step '{self.spec.id}' requires 'source' and 'output'.")
        if mapping is None:
            raise ValueError(f"Mapping step '{self.spec.id}' missing 'mapping'.")
        series = frame.get(source)
        if series is None:
            warn("derive.simple.map", step=self.spec.id, source=source, reason="missing_source")
            return frame
        mapped = series.map(mapping).fillna(default)
        df = frame.copy()
        df[output] = mapped
        log("derive.simple.map", step=self.spec.id, source=source, output=output)
        return df

    def _run_extract_street(self, frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
        source = cfg.get("source")
        output = cfg.get("output", "street")
        config = cfg.get("config") if isinstance(cfg.get("config"), Mapping) else {}
        if source is None:
            raise ValueError(f"Street extraction step '{self.spec.id}' requires 'source'.")
        if source not in frame.columns:
            warn("derive.simple.street", step=self.spec.id, source=source, reason="missing_source")
            return frame
        df = frame.copy()
        df[output] = frame[source].apply(lambda val: legacy.extract_street(val, config or {}))
        log("derive.simple.street", step=self.spec.id, source=source, output=output)
        return df

    def _run_month_index(self, frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
        """Derive month index from month column.
            The month index is a continuous count of months from a base month.
            args:
                month_col: Name of the source column containing month values (1-12).
                output: Name of the output column to store the month index.
                offset: (optional) Integer offset to add to the month index.
                fill_value: (optional) Value to use for missing or invalid month values.
        """
        config = {
            "enabled": True,
            "month_col": cfg.get("source") or cfg.get("config", {}).get("month_col") or 
                         cfg.get("month_col"),
            "output": cfg.get("output") or cfg.get("config", {}).get("output"),
        }
        
        if "config" in cfg and isinstance(cfg["config"], Mapping):
            config.update({k: v for k, v in cfg["config"].items() if k not in {"month_col", "output"}})
        else:
            for key in ("offset", "fill_value"):
                if key in cfg:
                    config[key] = cfg[key]
        if not config.get("month_col"):
            raise ValueError(f"Month index step '{self.spec.id}' missing 'source' or 'month_col'.")
        if not config.get("output"):
            config["output"] = "month_id"
        df = legacy.derive_month_index(frame.copy(), config)
        return df

    def _run_month_cyclical(self, frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
        out_prefix = None
        if isinstance(cfg.get("output"), str):
            out_prefix = cfg["output"]
        elif isinstance(cfg.get("output"), (list, tuple)):
            out_prefix = cfg.get("config", {}).get("out_prefix") or cfg.get("source") or "saleMonth"
        else:
            out_prefix = cfg.get("config", {}).get("out_prefix") or cfg.get("source") or "saleMonth"

        config = {
            "enabled": True,
            "month_col": cfg.get("source")
            or cfg.get("config", {}).get("month_col")
            or cfg.get("month_col"),
            "out_prefix": out_prefix,
        }
        if isinstance(cfg.get("config"), Mapping):
            config.update(cfg["config"])
        df = legacy.derive_month_cyclical(frame.copy(), config)
        outputs = cfg.get("output")
        if outputs and isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            a, b = outputs
            prefix = config.get("out_prefix", cfg.get("source") or "saleMonth")
            df = df.rename(columns={f"{prefix}_sin": a, f"{prefix}_cos": b})
        return df

    def _run_property_age(self, frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
        config = {
            "enabled": True,
            "year_built_col": cfg.get("source") or cfg.get("config", {}).get("year_built_col") or cfg.get("year_built_col"),
            "sale_year_col": cfg.get("config", {}).get("sale_year_col") or cfg.get("sale_year_col", "saleYear"),
            "output": cfg.get("output", "propertyAge"),
        }
        if isinstance(cfg.get("config"), Mapping):
            config.update(cfg["config"])
        return legacy.derive_property_age(frame.copy(), config)

    def _run_date_parts(self, frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
        source = cfg.get("source") or cfg.get("config", {}).get("source") or "saleDate"
        config = cfg.get("config") if isinstance(cfg.get("config"), Mapping) else {}
        year_output = cfg.get("year_output") or config.get("year_output") or "saleYear"
        month_output = cfg.get("month_output") or config.get("month_output") or "saleMonth"
        year_month_output = (
            cfg.get("year_month_output")
            or config.get("year_month_output")
            or "saleYearMonth"
        )
        if source not in frame.columns:
            warn("derive.date_parts", step=self.spec.id, source=source, reason="missing_source")
            return frame

        df = frame.copy()
        try:
            sale_dates = pd.to_datetime(df[source], errors="coerce")
        except Exception as exc:  # pragma: no cover - passthrough to caller
            warn("derive.date_parts", step=self.spec.id, reason="parse_error", error=str(exc))
            return frame

        if not sale_dates.isna().all():
            df[year_output] = sale_dates.dt.year
            df[month_output] = sale_dates.dt.month
            df[year_month_output] = sale_dates.dt.year * 100 + sale_dates.dt.month
            log(
                "derive.date_parts",
                step=self.spec.id,
                source=source,
                outputs=[year_output, month_output, year_month_output],
            )
        else:
            warn("derive.date_parts", step=self.spec.id, source=source, reason="all_nan")
        return df

    def _run_distance_to_cbd(self, frame: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
        output = cfg.get("output", "distance_to_cbd")
        lat_col = cfg.get("config", {}).get("latitude_col", "latitude")
        lon_col = cfg.get("config", {}).get("longitude_col", "longitude")

        if lat_col not in frame.columns or lon_col not in frame.columns:
            warn("derive.distance_to_cbd", step=self.spec.id, reason="missing_lat_lon_columns")
            return frame

        df = frame.copy()
        df[output] = _distance_to_cbd(df[lat_col], df[lon_col])
        log("derive.distance_to_cbd", step=self.spec.id, output=output)
        return df


__all__ = ["SimpleStep"]
