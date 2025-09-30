"""Typed configuration helpers for feature selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from property_adviser.core.config import load_config


def _merge_dicts(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class IdLikeConfig:
    enable: bool = True
    min_unique_ratio: float = 0.98
    drop_regex: tuple[str, ...] = ()


@dataclass(frozen=True)
class RedundancyConfig:
    enable: bool = True
    threshold: float = 0.95
    prefer_keep: tuple[str, ...] = ()


@dataclass(frozen=True)
class EliminationConfig:
    enable: bool = False
    estimator: str = "RandomForestRegressor"
    estimator_params: Mapping[str, Any] = field(default_factory=dict)
    step: int = 1
    min_features: int = 5
    scoring: str = "r2"
    cv: int = 3
    n_jobs: Optional[int] = None
    max_features: Optional[int] = None
    sample_rows: Optional[int] = None
    random_state: Optional[int] = None


@dataclass(frozen=True)
class FeatureSelectionConfig:
    input_file: Path
    output_dir: Path
    target: str
    correlation_threshold: Optional[float]
    use_top_k: Optional[bool]
    top_k: Optional[int]
    dataset_format: str
    exclude_columns: tuple[str, ...]
    mi_random_state: int
    id_like: IdLikeConfig
    redundancy: RedundancyConfig
    families: Mapping[str, Any]
    elimination: EliminationConfig
    scores_filename: str = "feature_scores.parquet"

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_path: Optional[Path] = None,
        config_dir: Optional[Path] = None,
    ) -> "FeatureSelectionConfig":
        base = base_path or Path.cwd()
        cfg_dir = config_dir or base
        project_root = cfg_dir.parent if cfg_dir.name == "config" else base

        def resolve(path_value: Any) -> Path:
            path = Path(path_value)
            if path.is_absolute():
                return path
            parts = path.parts
            if cfg_dir and parts and parts[0] == cfg_dir.name:
                anchor = cfg_dir.parent
                return (anchor / path) if anchor else (cfg_dir / path)
            return project_root / path

        if "input_file" not in mapping:
            raise KeyError("Feature selection config requires 'input_file'.")
        input_file = resolve(mapping["input_file"])  # type: ignore[index]
        output_dir = resolve(mapping.get("output_dir", "data/training"))
        target = mapping.get("target")
        if not target:
            raise KeyError("Feature selection config requires 'target'.")
        mi_random_state = int(mapping.get("mi_random_state", 7))

        correlation_threshold = mapping.get("correlation_threshold")
        if correlation_threshold is not None:
            correlation_threshold = float(correlation_threshold)

        use_top_k = mapping.get("use_top_k")
        if use_top_k is not None:
            use_top_k = bool(use_top_k)

        top_k = mapping.get("top_k")
        if top_k is not None:
            top_k = int(top_k)

        dataset_format = str(mapping.get("dataset_format", "csv")).lower()
        if dataset_format not in {"csv", "parquet"}:
            raise ValueError("dataset_format must be 'csv' or 'parquet'")

        exclude_columns = tuple(mapping.get("exclude_columns", []))

        id_like_cfg = mapping.get("id_like", {}) or {}
        id_like = IdLikeConfig(
            enable=bool(id_like_cfg.get("enable", True)),
            min_unique_ratio=float(id_like_cfg.get("min_unique_ratio", 0.98)),
            drop_regex=tuple(id_like_cfg.get("drop_regex", [])),
        )

        redundancy_cfg = mapping.get("redundancy", {}) or {}
        redundancy = RedundancyConfig(
            enable=bool(redundancy_cfg.get("enable", True)),
            threshold=float(redundancy_cfg.get("threshold", 0.95)),
            prefer_keep=tuple(redundancy_cfg.get("prefer_keep", [])),
        )

        elimination_cfg = mapping.get("elimination", {}) or {}
        elimination = EliminationConfig(
            enable=bool(elimination_cfg.get("enable", False)),
            estimator=str(elimination_cfg.get("estimator", "RandomForestRegressor")),
            estimator_params=elimination_cfg.get("estimator_params", {}) or {},
            step=int(elimination_cfg.get("step", 1)),
            min_features=int(elimination_cfg.get("min_features", 5)),
            scoring=str(elimination_cfg.get("scoring", "r2")),
            cv=int(elimination_cfg.get("cv", 3)),
            n_jobs=elimination_cfg.get("n_jobs"),
            max_features=(
                int(elimination_cfg["max_features"])
                if elimination_cfg.get("max_features") is not None
                else None
            ),
            sample_rows=(
                int(elimination_cfg["sample_rows"])
                if elimination_cfg.get("sample_rows") is not None
                else None
            ),
            random_state=(
                int(elimination_cfg["random_state"])
                if elimination_cfg.get("random_state") is not None
                else None
            ),
        )

        families = mapping.get("families", {}) or {}

        scores_filename = str(mapping.get("scores_filename", "feature_scores.parquet"))

        return cls(
            input_file=input_file,
            output_dir=output_dir,
            target=target,
            correlation_threshold=correlation_threshold,
            use_top_k=use_top_k,
            top_k=top_k,
            dataset_format=dataset_format,
            exclude_columns=exclude_columns,
            mi_random_state=mi_random_state,
            id_like=id_like,
            redundancy=redundancy,
            families=families,
            elimination=elimination,
            scores_filename=scores_filename,
        )


def load_feature_selection_config(path: Path) -> List[FeatureSelectionConfig]:
    raw = load_config(path)
    cfg_dir = path.parent
    base = cfg_dir.parent if cfg_dir.name == "config" else cfg_dir

    targets_cfg = raw.get("targets")
    if not targets_cfg:
        return [FeatureSelectionConfig.from_mapping(raw, base_path=base, config_dir=cfg_dir)]

    base_output_dir = raw.get("base_output_dir") or raw.get("output_dir", "data/training")
    configs: List[FeatureSelectionConfig] = []

    for target_cfg in targets_cfg:
        if not isinstance(target_cfg, Mapping):
            raise ValueError("Each target configuration must be a mapping.")

        merged = _merge_dicts(dict(raw), target_cfg)
        target_name = merged.pop("name", None) or merged.get("target")
        if not target_name:
            raise KeyError("Target configuration requires a 'name'.")
        merged["target"] = target_name

        output_subdir = target_cfg.get("output_subdir") or target_name
        merged["output_dir"] = str(Path(base_output_dir) / output_subdir)

        configs.append(
            FeatureSelectionConfig.from_mapping(
                merged,
                base_path=base,
                config_dir=cfg_dir,
            )
        )

    return configs


__all__ = [
    "FeatureSelectionConfig",
    "IdLikeConfig",
    "RedundancyConfig",
    "EliminationConfig",
    "load_feature_selection_config",
]
