import json
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from property_adviser.predict import feature_store as feature_store_mod
from property_adviser.predict import model_prediction as predict_mod


@pytest.fixture()
def temp_environment(monkeypatch, tmp_path):
    # Prepare directory skeleton
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    training_dir = tmp_path / "data" / "training"
    training_dir.mkdir(parents=True)

    # Fake feature metadata
    metadata = {
        "target": "salePrice",
        "feature_metadata": {
            "timestamp": "20250101_000000",
            "month_column": "saleYearMonth",
            "model_input_columns": [
                "saleYearMonth",
                "bed",
                "bath",
                "car",
                "landSizeM2",
                "floorSizeM2",
                "propertyAge",
                "propertyAgeBand",
                "propertyType",
                "suburb_price_median_current",
                "count",
                "std",
                "suburb_price_median_3m",
                "suburb_price_median_6m",
                "suburb_price_median_12m",
                "suburb_txn_count_3m",
                "suburb_txn_count_6m",
                "suburb_txn_count_12m",
                "suburb_volatility_3m",
                "suburb_volatility_6m",
                "suburb_volatility_12m",
                "suburb_delta_3m",
                "suburb_delta_12m",
                "rel_price_vs_suburb_median",
                "saleType",
                "agency",
                "landUse",
                "developmentZone",
                "ownerType",
                "agencyBrand",
            ],
            "numeric_features": [
                "saleYearMonth",
                "bed",
                "bath",
                "car",
                "landSizeM2",
                "floorSizeM2",
                "propertyAge",
                "suburb_price_median_current",
                "count",
                "std",
                "suburb_price_median_3m",
                "suburb_price_median_6m",
                "suburb_price_median_12m",
                "suburb_txn_count_3m",
                "suburb_txn_count_6m",
                "suburb_txn_count_12m",
                "suburb_volatility_3m",
                "suburb_volatility_6m",
                "suburb_volatility_12m",
                "suburb_delta_3m",
                "suburb_delta_12m",
                "rel_price_vs_suburb_median",
            ],
            "categorical_features": [
                "propertyAgeBand",
                "propertyType",
                "saleType",
                "agency",
                "landUse",
                "developmentZone",
                "ownerType",
                "agencyBrand",
            ],
            "impute": {
                "numeric": {
                    "bed": 3,
                    "bath": 2,
                    "car": 1,
                    "landSizeM2": 400.0,
                    "floorSizeM2": 180.0,
                    "propertyAge": 20.0,
                    "saleYearMonth": 202501,
                },
                "categorical": {
                    "propertyType": "House",
                    "propertyAgeBand": "6-20",
                },
            },
            "property_age": {
                "bands": [5, 20],
                "labels": ["0-5", "6-20", "21+"],
            },
        },
    }

    meta_path = training_dir / "feature_metadata.json"
    meta_path.write_text(json.dumps(metadata))

    # Dummy model
    dummy_model = MagicMock()
    dummy_model.predict.return_value = np.array([1_000_000.0])
    joblib.dump(dummy_model, models_dir / "best_model.joblib")

    # Point config paths to temporary dirs
    monkeypatch.setattr(predict_mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(predict_mod, "TRAINING_DIR", training_dir)

    reference_row = pd.Series(
        {
            "suburb_price_median_current": 1_050_000,
            "count": 20,
            "std": 100_000,
            "suburb_price_median_3m": 1_020_000,
            "suburb_price_median_6m": 1_000_000,
            "suburb_price_median_12m": 950_000,
            "suburb_txn_count_3m": 15,
            "suburb_txn_count_6m": 30,
            "suburb_txn_count_12m": 60,
            "suburb_volatility_3m": 110_000,
            "suburb_volatility_6m": 120_000,
            "suburb_volatility_12m": 130_000,
            "suburb_delta_3m": 0.05,
            "suburb_delta_12m": 0.12,
            "rel_price_vs_suburb_median": 1.1,
            "saleType": "Auction",
            "agency": "Example Agency",
            "landUse": "Detached Dwelling",
            "developmentZone": "RZ",
            "ownerType": "Owner Occupied",
            "agencyBrand": "Example Brand",
        }
    )

    def fake_fetch(suburb: str, sale_year_month: int, columns):
        return reference_row.reindex(columns)

    monkeypatch.setattr(feature_store_mod, "fetch_reference_features", fake_fetch)

    return dummy_model


def test_prepare_prediction_data_imputes_missing_values(temp_environment):
    _, metadata = predict_mod.load_trained_model()

    rows = [
        {
            "saleYearMonth": 202506,
            "suburb": "Mitcham",
            "bed": 4,
            "bath": 3,
            "car": 2,
            "landSize": 500,
            "floorSize": 220,
            "propertyType": "House",
            "yearBuild": 2000,
        },
        {
            # Missing several optional fields – should be imputed
            "saleYearMonth": 202501,
            "suburb": "Mitcham",
            "bed": 3,
            "bath": 2,
            "car": 1,
            "propertyType": "Unit",
        },
    ]

    prepared = predict_mod._prepare_prediction_data(rows, metadata)

    # Shape matches expected model columns
    assert list(prepared.columns) == metadata["feature_metadata"]["model_input_columns"]
    assert prepared.shape[0] == 2

    # Derived features present
    assert "propertyAge" in prepared.columns
    assert "propertyAgeBand" in prepared.columns

    # Imputation applied (no NaNs for required columns)
    assert prepared.isna().sum().sum() == 0


def test_predict_property_price_uses_model(temp_environment):
    dummy_model = temp_environment
    price = predict_mod.predict_property_price(
        yearmonth=202506,
        bed=3,
        bath=2,
        car=1,
        property_type="House",
        street="Example",
        suburb="Mitcham",
        land_size=450,
        floor_size=190,
        year_built=1999,
    )
    assert price == pytest.approx(1_000_000.0)
    dummy_model.predict.assert_called_once()


def test_batch_prediction_preserves_order(temp_environment):
    properties = [
        {
            "saleYearMonth": 202506,
            "suburb": "Mitcham",
            "bed": 3,
            "bath": 2,
            "car": 1,
            "propertyType": "House",
        },
        {
            "saleYearMonth": 202507,
            "suburb": "Blackburn",
            "bed": 2,
            "bath": 1,
            "car": 0,
            "propertyType": "Unit",
        },
    ]

    preds = predict_mod.predict_property_prices_batch(properties)
    assert preds == [pytest.approx(1_000_000.0), pytest.approx(1_000_000.0)]
