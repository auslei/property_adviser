"""Compatibility wrappers for the training pipeline."""
from __future__ import annotations

import json
from datetime import datetime
import math
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

        best_overall: Optional[Dict[str, Any]] = None
        best_score = float("-inf")

        entries: list[Dict[str, Any]] = []
        for res in results:
            scores_records = res.scores_table.to_dict(orient="records")
            entry = {
                "name": res.name,
                "target": res.target,
                "forecast_window": res.forecast_window,
                "timestamp": res.timestamp,
                "best_model": res.best_model,
                "best_model_path": str(res.best_model_path),
                "canonical_model_path": str(res.canonical_model_path),
                "summary_path": str(res.summary_path),
                "scores_path": str(res.scores_path),
                "metadata_path": str(res.metadata_path),
                "validation_month": res.validation_month,
                "scores": scores_records,
                "duration_seconds": res.duration,
            }
            entries.append(entry)

            best_row = next((row for row in scores_records if row.get("model") == res.best_model), None)
            if not best_row:
                continue
            score = best_row.get("val_r2")
            if score is None or not isinstance(score, (int, float)):
                continue
            score_value = float(score)
            if math.isnan(score_value):
                continue
            if score_value > best_score:
                best_score = score_value

                rmse_val = best_row.get("val_rmse")
                rmse_value: Optional[float] = None
                if isinstance(rmse_val, (int, float)):
                    rmse_f = float(rmse_val)
                    if not math.isnan(rmse_f):
                        rmse_value = rmse_f

                mae_val = best_row.get("val_mae")
                mae_value: Optional[float] = None
                if isinstance(mae_val, (int, float)):
                    mae_f = float(mae_val)
                    if not math.isnan(mae_f):
                        mae_value = mae_f

                best_overall = {
                    "target_name": res.name,
                    "target": res.target,
                    "forecast_window": res.forecast_window,
                    "model": res.best_model,
                    "timestamp": res.timestamp,
                    "val_r2": score_value,
                    "val_rmse": rmse_value,
                    "val_mae": mae_value,
                    "best_model_path": str(res.best_model_path),
                    "canonical_model_path": str(res.canonical_model_path),
                    "summary_path": str(res.summary_path),
                    "scores_path": str(res.scores_path),
                }

        report_path = None
        if results:
            report_dir = Path(results[0].canonical_model_path).parents[1]
            report_dir.mkdir(parents=True, exist_ok=True)
            report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"training_report_{report_timestamp}.json"
            payload = {
                "generated_at": report_timestamp,
                "targets": entries,
                "best_overall": best_overall,
            }
            report_path.write_text(json.dumps(payload, indent=2))
            log("train.report_saved", path=str(report_path))

        return {
            "targets": entries,
            "best_overall": best_overall,
            "report_path": str(report_path) if report_path else None,
        }
    except Exception as exc:
        log_exc("train.error", exc)
        raise


__all__ = [
    "MODEL_FACTORY",
    "train_timeseries_model",
]
