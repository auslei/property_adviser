# Prediction Module Agent Guide

## Purpose & Scope
- Serve deterministic inference for trained property price models by loading persisted pipelines, reconstructing inputs, and emitting predictions and confidence intervals.
- Present a minimal surface area (`predict_property_price`, batch helpers, CLI integrations) that downstream apps can rely on.
- Leverage `property_adviser.core` for logging, IO, and configuration to keep behaviour aligned with training.

## Design Commitments
- **Clear interface**: Public APIs accept explicit parameters (property dictionaries, feature metadata paths) and return plain Python objects; no hidden globals.
- **High cohesion**: Data preparation, metadata loading, and scoring live within this package. Presentation layers call into these helpers rather than duplicating logic.
- **Low coupling**: Prediction depends on artefacts produced by training (`best_model.joblib`, `model_scores_*.csv`, `feature_metadata.json`) without importing training internals.
- **Reusability**: Common data preparation utilities are candidates for promotion to `core` when shared across modules.

## Key Components
- `model_prediction.py`: loads bundles, prepares feature frames, performs single/batch predictions, and exposes confidence estimates.
- `feature_store.py`: provides cached access to derived datasets for suburb/street lookups used by apps.

## Required Artefacts
- `models/best_model.joblib` (or a specific day's `models/<YYYYMMDD>/<target>/best_model.joblib`)
- `data/training/feature_metadata.json` (promoted copy) or per-target `data/features/<target>/feature_metadata.json`
- Latest `models/<YYYYMMDD>/<target>/model_scores.csv` for RMSE-driven confidence intervals

## Public APIs
```python
from property_adviser.predict.model_prediction import (
    load_trained_model,
    predict_property_price,
    predict_property_prices_batch,
    predict_with_confidence_interval,
)
```
- `load_trained_model(path=None)` returns `(pipeline, metadata)`; default path is `models/best_model.joblib`.
- `predict_property_price(...)` scores a single property, deriving helper fields (`saleYear`, `propertyAge`, etc.) from inputs.
- `predict_property_prices_batch(properties)` accepts a list of dictionaries and returns predictions ordered like the input.
- `predict_with_confidence_interval(...)` augments predictions with a ± interval based on validation RMSE.

## Input Expectations
Each property dict should provide:
- `saleYearMonth` (YYYYMM)
- `suburb`
- `bed`, `bath`, `car`
- `propertyType`
Optional fields: `landSize`, `floorSize`, `yearBuilt`, `street`.

Helpers derive seasonal features, property age, and suburb-level aggregates using cached feature-store lookups. Missing values are imputed with medians/modes captured during training.

## Logging & Error Handling
- Structured logs (`predict.load_model`, `predict.prepare_frame`, `predict.score`) use `core.app_logging`.
- Errors surface informative messages when artefacts are missing; caller can catch and surface in UIs.

## Maintenance Checklist
1. Update this guide whenever the bundle schema or required inputs change.
2. When introducing new engineered features, ensure preprocessing metadata contains the information needed to recreate them at inference time.
3. Keep batch helpers deterministic—ordering of outputs must match inputs for UI alignment.
