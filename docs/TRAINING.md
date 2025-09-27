## Training Module — Documentation

This module handles the **model training stage** of Property Adviser. It consumes the
prepared feature matrix (`X`), target values (`y`), and optional feature selection
metadata to produce timestamped model artifacts and score summaries.

### Structure
```
property_adviser/
  train/
    cli.py            # CLI entry point (pa-train)
    model_training.py # grid-search orchestration + persistence helpers
```

### Data Inputs
- **Features (`X`)**: tabular dataset, typically produced by the feature selection
  module (`feature_scores` selections applied).
- **Target (`y`)**: file with the target column specified in the config (`salePrice`
  by default).
- **Feature scores (optional)**: table from the feature selection step. If present,
  the training pipeline respects `selected`, `include`, and `exclude` flags to
  keep the feature set consistent with manual overrides.

### Configuration (`config/model.yml`)
- `task`: currently informational (defaults to `regression`).
- `target`: name of the column to model.
- `log_target`: if `true`, train on `log1p(salePrice)` and predict with `expm1(...)`. Rows with
  non-positive targets are dropped automatically and logged.
- `input`:
  - `path`: base directory for the training inputs.
  - `X`, `y`: filenames (relative to `input.path` unless absolute).
  - `feature_scores`: optional file for manual feature overrides.
- `model_path.base`: directory for model artifacts (created if missing).
- `split`:
  - `validation_month`: month to hold out; the latest month is used if omitted.
  - `month_column`: column in `X` that stores the sale month (`saleYearMonth`).
- `models`: mapping of model names to `{enabled, grid}` definitions. Supported
  names are `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`,
  `RandomForestRegressor`, and `GradientBoostingRegressor`. If the section is
  missing, all supported models are enabled with empty grids.
  - Grid keys may omit the `model__` prefix (`alpha` → `model__alpha`). Keys for
    preprocessing parameters can use the `preprocessor__` prefix directly.

### Workflow
1. Load configuration (plus any overrides from CLI/programmatic calls) and set up
   logging (`train.start`).
2. Read `X`, `y`, and optionally `feature_scores` (Parquet and CSV are supported).
3. Apply feature overrides: `selected`, `include`, and `exclude` columns in the
   scores table dictate which predictors remain.
4. Determine the validation month: use the requested value if available; otherwise
   fall back to the most recent month present in `X`.
5. Split the dataset by month (`train.split`): training uses all rows before the
   validation month; validation uses the held-out month. The month column is
   dropped from both splits before modelling.
6. Build a preprocessing pipeline once using the training predictors: numeric
   columns receive a median imputer + standard scaler; categorical columns receive
   a most-frequent imputer + one-hot encoder (`sparse_output=False`).
7. For each enabled model:
   - Wrap the preprocessor and estimator in a `Pipeline`.
   - If `log_target` is enabled, the pipeline is wrapped in a
     `TransformedTargetRegressor(log1p ↔︎ expm1)` so CV scoring/evaluation stay in
     the original currency scale.
   - Run `GridSearchCV` with 3-fold cross-validation optimising R² (`train.gridsearch`).
   - Predict on the validation month and log MAE/RMSE/R² (`train.validation`).
8. Select the model with the highest validation R², capture best params, and
   persist artifacts.
9. Save outputs:
   - `best_model_<name>_<timestamp>.joblib`: serialized dict containing the fitted
     pipeline, metadata (target, month column, validation month, feature lists,
     best params, models tried).
   - `best_model.joblib` + `best_model.json`: canonical copies for downstream
     prediction services (JSON includes summary metrics, params, and the timestamped path).
   - `model_scores_<timestamp>.csv`: validation metrics and CV scores for each
     candidate (`train.save_scores`).
   - `feature_metadata.json`: feature lists and simple imputation defaults used by
     the prediction utilities.
10. Return a dictionary usable by the GUI: best model name/path, scores path,
    validation month, and full score table.

### CLI Usage
```bash
uv run pa-train --config config/model.yml --verbose
```
- `--config`: optional override; defaults to `config/model.yml`.
- `--verbose`: enables more detailed logging.

### Programmatic Usage
```python
from property_adviser.train.model_training import train_timeseries_model

result = train_timeseries_model(
    config_path="config/model.yml",
    overrides={"verbose": True, "split": {"validation_month": "2025-05"}}
)

print(result["best_model"], result["validation_month"])
print(result["best_model_path"], result["scores_path"])
```

### Extending / Customising
- **Adding models**: register the estimator in `MODEL_FACTORY` and add a matching
  entry in `models` with `enabled: true`.
- **Adjusting preprocessing**: modify `_build_preprocessor` in
  `model_training.py` (e.g., add custom transformers) and expose parameters via
  the grid with `preprocessor__...` keys.
- **Custom validation**: `_choose_validation_month` and `_split_by_month` are the
  touch points for alternative splitting strategies.

### Logging
Key events emitted to the structured logger:
- `train.start`, `train.feature_scores_loaded`
- `train.log_target_drop_nonpositive`
- `train.validation_month`, `train.split`, `train.preprocessor`
- `train.gridsearch`, `train.validation`
- `train.save_model`, `train.save_scores`
- `train.error` (with exception details)

These logs provide traceability for experiments and aid in debugging pipeline
changes.
