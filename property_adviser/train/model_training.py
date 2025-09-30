"""Compatibility wrappers for the training pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from property_adviser.core.app_logging import log_exc, setup_logging
from property_adviser.train.config import TrainingConfig, load_training_config
from property_adviser.train.pipeline import MODEL_FACTORY, TrainingResult, run_training


def train_timeseries_model(
    config_path: Optional[str] = "model.yml",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Backward-compatible entry point used by CLI and external callers."""
    overrides = overrides or {}
    initial_verbose = bool(overrides.get("verbose", False))
    setup_logging(initial_verbose)

    cfg_path = Path(config_path) if config_path else None

    try:
        config: TrainingConfig = load_training_config(cfg_path, overrides)
        setup_logging(config.verbose)
        result: TrainingResult = run_training(config)
        return {
            "best_model": result.best_model,
            "best_model_path": str(result.best_model_path),
            "canonical_model_path": str(result.canonical_model_path),
            "summary_path": str(result.summary_path),
            "scores_path": str(result.scores_path),
            "validation_month": result.validation_month,
            "scores": result.scores_table.to_dict(orient="records"),
            "duration_seconds": result.duration,
        }
    except Exception as exc:
        log_exc("train.error", exc)
        raise


__all__ = [
    "MODEL_FACTORY",
    "train_timeseries_model",
]
