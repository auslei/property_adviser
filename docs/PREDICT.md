## Prediction Module — Documentation

This module serves the final stage of the Property Adviser pipeline. It loads the
latest persisted model bundle, reconstructs the feature frame expected by the
pipeline, and returns sale-price predictions for one or more properties.

### Key APIs (`property_adviser/predict/model_prediction.py`)
- `load_trained_model()`
  - Returns the fitted model bundle and feature metadata.
  - Searches for `models/best_model.joblib` (falls back to `best_model.pkl`).
- `predict_property_price(...)`
  - Convenience wrapper for scoring a single property. Accepts the canonical
    fields and the optional geometry/age inputs, then defers to the batch helper.
- `predict_property_prices_batch(properties)`
  - Scores a list of property dictionaries. Each dictionary should provide the
    minimal required fields described below; optional fields (e.g. land size)
    are safely imputed if absent.
- `predict_with_confidence_interval(...)`
  - Wraps `predict_property_price` and attaches a symmetric interval using the
    stored validation RMSE as a proxy for model uncertainty.

### Persisted Metadata (`data/training/feature_metadata.json`)
`train_timeseries_model` now writes an accompanying JSON file containing:
- `feature_metadata.model_input_columns`: the exact column order passed to the
  estimator during training. This is what `_prepare_prediction_data` rebuilds.
- `feature_metadata.numeric_features` / `categorical_features`: column lists used
  to restore dtypes before invoking the pipeline.
- `feature_metadata.impute.numeric` / `categorical`: simple median/mode values
  captured from the training frame. These populate missing predictors during
  prediction.
- `property_age.bands` / `labels`: the bucket edges and labels used to recreate
  the `propertyAgeBand` categorical feature.
- High-level context (`target`, `validation_month`, `models_considered`,
  `selected_model`) for auditability.

### Required Prediction Inputs
Each property dictionary must provide at least:
- `saleYearMonth`: integer/str in YYYYMM form (e.g. `202506`).
- `suburb`: suburb name (case-insensitive; used to retrieve suburb-level features).
- `bed`, `bath`, `car`: numeric counts of bedrooms/bathrooms/car spaces.
- `propertyType`: string matching the categories seen during training.
- `street`: optional visually, but surfaced in the UI for completeness.

Optional fields that improve accuracy when present:
- `landSize` (square metres)
- `floorSize` (square metres)
- `yearBuild` / `yearBuilt`

`_prepare_prediction_data` normalises these names and derives:
- `saleYear`, `saleMonth` (from `saleYearMonth`).
- `propertyAge` (if `yearBuilt` is available) and `propertyAgeBand` using the
  stored buckets.
- Suburb-level aggregates (`suburb_price_median_*`, transaction counts, etc.) are
  looked up from the derived dataset based on `suburb` and `saleYearMonth`.

### Imputation & Type Handling
- Numeric features: coerced via `pd.to_numeric(..., errors="coerce")` and filled
  with the cached medians. Unseen columns fall back to zero if no metadata value
  exists.
- Categoricals: cast to `object` and filled with stored modes (default `"Unknown"`).
- `propertyAgeBand`: always converted to a plain string label to avoid category
  mismatches, then imputed with `Unknown` if necessary.

### Example Usage
```python
from property_adviser.predict.model_prediction import predict_property_price

estimate = predict_property_price(
    yearmonth=202506,
    bed=3,
    bath=2,
    car=1,
    property_type="House",
    street="Example Street",
    suburb="Mitcham",
    land_size=450.0,
    floor_size=180.0,
    year_built=1998,
)
print(f"Estimated sale price: ${estimate:,.0f}")
```

For batch scoring, supply a list of similarly structured dictionaries to
`predict_property_prices_batch`. The helper returns a list of floats ordered the
same as the input records.

### Validation & Confidence Estimates
`predict_with_confidence_interval` pulls the best validation RMSE from
`models/model_metrics.json` (if present) and constructs a 95% interval using the
normal approximation. The centre value is identical to
`predict_property_price(...)`.

### Maintenance Tips
- Regenerate `feature_metadata.json` whenever you retrain models so prediction
  stays aligned with the latest feature set.
- If you introduce new engineered features during preprocessing/selection,
  ensure they can be recreated here from the available raw inputs or extend the
  prediction interface accordingly.
- When deploying outside notebooks, keep `models/best_model.joblib` and
  `data/training/feature_metadata.json` together—they are both required for
  deterministic predictions.
