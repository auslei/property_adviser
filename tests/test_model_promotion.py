import json
from pathlib import Path

import pytest

from property_adviser.train.promotion import promote_models


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_promote_models_activates_single_target(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    training_dir = tmp_path / "data" / "training"

    source_model_dir = models_dir / "price_future_12m"
    source_model = source_model_dir / "best_model.joblib"
    source_summary = source_model_dir / "best_model.json"
    source_scores = source_model_dir / "model_scores.csv"
    metadata_path = training_dir / "price_future_12m" / "feature_metadata.json"

    _write(source_model, "model-bytes")
    _write(source_summary, json.dumps({"target": "price_future_12m"}))
    _write(source_scores, "model,score\nGradientBoostingRegressor,0.97\n")
    _write(metadata_path, json.dumps({"model_input_columns": ["a", "b"]}))

    report_payload = {
        "generated_at": "20250101_000000",
        "targets": [
            {
                "name": "price_future_12m",
                "target": "price_future_12m_smooth_delta",
                "timestamp": "20250101_000000",
                "best_model": "GradientBoostingRegressor",
                "canonical_model_path": str(source_model.relative_to(tmp_path)),
                "summary_path": str(source_summary.relative_to(tmp_path)),
                "scores_path": str(source_scores.relative_to(tmp_path)),
                "metadata_path": str(metadata_path),
            }
        ],
        "best_overall": {
            "target_name": "price_future_12m",
            "target": "price_future_12m_smooth_delta",
        },
    }
    report_path = models_dir / "training_report_20250101_000000.json"
    _write(report_path, json.dumps(report_payload))

    from property_adviser.train import promotion as promotion_module

    monkeypatch.setattr(promotion_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(promotion_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(promotion_module, "TRAINING_DIR", training_dir)

    result = promote_models(report_path=report_path, copy_scores=True)

    dest_dir = models_dir / "model_final" / "price_future_12m"
    assert (dest_dir / "best_model.joblib").exists()
    assert json.loads((dest_dir / "promotion_manifest.json").read_text())[
        "source_model"
    ].endswith("best_model.joblib")
    active_bundle = models_dir / "model_final" / "best_model.joblib"
    assert active_bundle.exists()

    promoted_metadata = training_dir / "feature_metadata.json"
    assert promoted_metadata.exists()

    assert result["activated_target"] == "price_future_12m"
    assert result["promotions"][0]["target_name"] == "price_future_12m"


def test_promote_models_best_per_window(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    training_dir = tmp_path / "data" / "training"

    # Create two candidate models for the same forecast window
    source_a = models_dir / "price_future_12m_alpha"
    model_a = source_a / "best_model.joblib"
    summary_a = source_a / "best_model.json"
    metadata_a = training_dir / "price_future_12m_alpha" / "feature_metadata.json"

    source_b = models_dir / "price_future_12m_beta"
    model_b = source_b / "best_model.joblib"
    summary_b = source_b / "best_model.json"
    metadata_b = training_dir / "price_future_12m_beta" / "feature_metadata.json"

    for file in (model_a, summary_a, model_b, summary_b, metadata_a, metadata_b):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("placeholder", encoding="utf-8")

    report_payload = {
        "generated_at": "20250101_000000",
        "targets": [
            {
                "name": "price_future_12m_alpha",
                "target": "price_future_12m_delta",
                "forecast_window": "12m",
                "timestamp": "20250101_000000",
                "best_model": "GradientBoostingRegressor",
                "canonical_model_path": str(model_a.relative_to(tmp_path)),
                "summary_path": str(summary_a.relative_to(tmp_path)),
                "scores_path": str((source_a / "scores.csv").relative_to(tmp_path)),
                "metadata_path": str(metadata_a),
                "scores": [
                    {
                        "model": "GradientBoostingRegressor",
                        "val_r2": 0.9,
                    }
                ],
            },
            {
                "name": "price_future_12m_beta",
                "target": "price_future_12m_delta",
                "forecast_window": "12m",
                "timestamp": "20250101_000001",
                "best_model": "RandomForestRegressor",
                "canonical_model_path": str(model_b.relative_to(tmp_path)),
                "summary_path": str(summary_b.relative_to(tmp_path)),
                "scores_path": str((source_b / "scores.csv").relative_to(tmp_path)),
                "metadata_path": str(metadata_b),
                "scores": [
                    {
                        "model": "RandomForestRegressor",
                        "val_r2": 0.4,
                    }
                ],
            },
        ],
    }
    report_path = models_dir / "training_report_20250101_000000.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    from property_adviser.train import promotion as promotion_module

    monkeypatch.setattr(promotion_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(promotion_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(promotion_module, "TRAINING_DIR", training_dir)

    result = promote_models(report_path=report_path, include_all_targets=True)

    assert len(result["promotions"]) == 1
    assert result["promotions"][0]["target_name"] == "price_future_12m_alpha"
    assert result["promotions"][0]["forecast_window"] == "12m"
