"""Compatibility wrappers for the training pipeline."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from property_adviser.core.app_logging import log, log_exc, setup_logging
from property_adviser.train.config import load_training_config
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
        configs = load_training_config(cfg_path, overrides)
        results: list[TrainingResult] = []

        for cfg in configs:
            if cfg.verbose:
                setup_logging(True)
            result = run_training(cfg)
            results.append(result)

        entries = [
            {
                "name": res.name,
                "target": res.target,
                "timestamp": res.timestamp,
                "best_model": res.best_model,
                "best_model_path": str(res.best_model_path),
                "canonical_model_path": str(res.canonical_model_path),
                "summary_path": str(res.summary_path),
                "scores_path": str(res.scores_path),
                "metadata_path": str(res.metadata_path),
                "validation_month": res.validation_month,
                "scores": res.scores_table.to_dict(orient="records"),
                "duration_seconds": res.duration,
            }
            for res in results
        ]

        report_path = None
        if results:
            report_dir = Path(results[0].canonical_model_path).parents[1]
            report_dir.mkdir(parents=True, exist_ok=True)
            report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"training_report_{report_timestamp}.json"
            payload = {
                "generated_at": report_timestamp,
                "targets": entries,
            }
            report_path.write_text(json.dumps(payload, indent=2))
            log("train.report_saved", path=str(report_path))

        return {
            "targets": entries,
            "report_path": str(report_path) if report_path else None,
        }
    except Exception as exc:
        log_exc("train.error", exc)
        raise


__all__ = [
    "MODEL_FACTORY",
    "train_timeseries_model",
]
