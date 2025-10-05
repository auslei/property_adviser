"""Typed configuration helpers for model training."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import re

from property_adviser.config import PROJECT_ROOT
from property_adviser.core.config import load_config


@dataclass(frozen=True)
class InputConfig:
    base: Path
    X: Path
    y: Path
    feature_scores: Optional[Path]


@dataclass(frozen=True)
class SplitConfig:
    validation_month: Optional[str]
    month_column: str


@dataclass(frozen=True)
class ModelConfig:
    enabled: bool
    grid: Dict[str, Any]


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    task: str
    target: str
    log_target: bool
    verbose: bool
    input: InputConfig
    split: SplitConfig
    artifacts_dir: Path
    models: Dict[str, ModelConfig]
    forecast_window: Optional[str]


def _merge_dicts(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _infer_forecast_window(name: str, target: str) -> Optional[str]:
    pattern = re.compile(r"(\d+)\s*(?:months?|m)(?![A-Za-z0-9])", re.IGNORECASE)
    for value in (name, target):
        if not value:
            continue
        match = pattern.search(value)
        if match:
            return f"{match.group(1)}m"
    return None


def _resolve_path(value: str, *, project_root: Path, config_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == config_dir.name:
        anchor = config_dir.parent
        return (anchor / path) if anchor else (config_dir / path)
    return project_root / path


def _build_training_config(
    raw: Mapping[str, Any],
    *,
    name: str,
    project_root: Path,
    config_dir: Path,
    append_name: bool,
) -> TrainingConfig:
    input_cfg = raw.get("input", {})
    # Feature selection artifacts now default under data/features
    base_dir = _resolve_path(input_cfg.get("path", "data/features"), project_root=project_root, config_dir=config_dir)
    base_dir = base_dir if base_dir.is_absolute() else base_dir.resolve()

    def resolve_under_base(value: Optional[Any]) -> Optional[Path]:
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return base_dir / path

    X_path = resolve_under_base(input_cfg.get("X", "X.parquet"))
    y_path = resolve_under_base(input_cfg.get("y", "y.parquet"))
    if X_path is None or y_path is None:
        raise ValueError("Model config must specify input.X and input.y paths.")
    feature_scores_path = resolve_under_base(input_cfg.get("feature_scores"))

    model_path_cfg = raw.get("model_path", {})
    artifacts_dir = _resolve_path(model_path_cfg.get("base", "models"), project_root=project_root, config_dir=config_dir)
    # Add a daily bucket (YYYYMMDD) so same-day runs overwrite within the day
    day_dir = datetime.now().strftime("%Y%m%d")
    artifacts_dir = artifacts_dir / day_dir
    append_flag = model_path_cfg.get("append_target", append_name)
    subdir = model_path_cfg.get("subdir")
    if append_flag:
        artifacts_dir = artifacts_dir / name
    if subdir:
        artifacts_dir = artifacts_dir / subdir

    split_cfg = raw.get("split", {})
    split = SplitConfig(
        validation_month=split_cfg.get("validation_month"),
        month_column=split_cfg.get("month_column", "saleYearMonth"),
    )

    forecast_window = raw.get("forecast_window")
    if isinstance(forecast_window, (int, float)):
        forecast_window = f"{int(forecast_window)}m"
    if forecast_window is None:
        inferred = _infer_forecast_window(str(raw.get("name") or name), str(raw.get("target") or name))
        forecast_window = inferred

    models_cfg = raw.get("models") or {
        model_name: {"enabled": True, "grid": {}}
        for model_name in (
            "LinearRegression",
            "Ridge",
            "Lasso",
            "ElasticNet",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "SARIMAX",
        )
    }
    models: Dict[str, ModelConfig] = {}
    for model_name, entry in models_cfg.items():
        enabled = bool(entry.get("enabled", True))
        grid = dict(entry.get("grid", {}))
        models[model_name] = ModelConfig(enabled=enabled, grid=grid)

    return TrainingConfig(
        name=name,
        task=raw.get("task", "regression"),
        target=raw.get("target", name),
        log_target=bool(raw.get("log_target", False)),
        verbose=bool(raw.get("verbose", False)),
        input=InputConfig(
            base=base_dir,
            X=X_path,
            y=y_path,
            feature_scores=feature_scores_path,
        ),
        split=split,
        artifacts_dir=artifacts_dir,
        models=models,
        forecast_window=forecast_window,
    )


def load_training_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> List[TrainingConfig]:
    cfg_path = config_path or (PROJECT_ROOT / "config" / "model.yml")
    config_dir = cfg_path.parent
    project_root = config_dir.parent if config_dir.name == "config" else config_dir
    raw = load_config(cfg_path)
    if overrides:
        raw = _merge_dicts(raw, overrides)

    targets_cfg = raw.get("targets")
    if targets_cfg:
        configs: List[TrainingConfig] = []
        for target_cfg in targets_cfg:
            if not isinstance(target_cfg, Mapping):
                raise ValueError("Each target entry must be a mapping.")
            merged = _merge_dicts(dict(raw), target_cfg)
            target_name = merged.pop("name", None) or merged.get("target")
            if not target_name:
                raise KeyError("Each target entry requires a 'name'.")
            merged["target"] = merged.get("target", target_name)
            configs.append(
                _build_training_config(
                    merged,
                    name=str(target_name),
                    project_root=project_root,
                    config_dir=config_dir,
                    append_name=True,
                )
            )
        return configs

    name = raw.get("name") or raw.get("target", "default")
    return [
        _build_training_config(
            raw,
            name=str(name),
            project_root=project_root,
            config_dir=config_dir,
            append_name=False,
        )
    ]


__all__ = [
    "InputConfig",
    "SplitConfig",
    "ModelConfig",
    "TrainingConfig",
    "load_training_config",
]
