# Property Analyst

## Overview
End-to-end pipeline for forecasting next-year property sale prices from suburb-level CSVs. It standardises and enriches inputs, learns a factor relative to suburb-month medians, trains multiple regressors, and surfaces artefacts in a Streamlit UI for exploration.

## Repository Layout
- `data/` – drop raw suburb-level CSV exports here (inputs).
- `data_preprocess/` – cleaned dataset (`cleaned.parquet`) and preprocessing metadata (`metadata.json`).
- `data_training/` – training matrices (`X.parquet`, `y.parquet`, split dumps) and feature metadata/importances.
- `models/` – fitted model artefacts and evaluation metrics.
- `config/` – YAML configurations for preprocessing, feature engineering, and model training.
- `src/` – preprocessing, suburb baselines, feature selection, model training, and pipeline runner.
- `app/` – Streamlit overview page and utilities.
- `app/pages/` – multi-page Streamlit workflows (Preprocessing, Feature Engineering, Model Selection).

## Environment Setup
- Python 3.10+ (CPython).
- Create a virtual environment and install deps:
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- Place CSVs in `data/` (UTF‑8 with BOM accepted). Multiple files are concatenated.

## Running the Pipeline
Run end-to-end (preprocess → baselines → features → training):
```bash
python -m src.run_pipeline
```
Force preprocessing even if raw file signatures (size/mtime) haven’t changed:
```bash
python -m src.run_pipeline --force
```

### Pipeline Stages
1. Preprocessing (`src/preprocess.py`)
   - Normalises column names, drops <35% non‑null columns, standardises strings.
   - Extracts `street` from `streetAddress`, simplifies `propertyType`, coerces numerics.
   - Reformats `saleDate` to `YYYYMM`, persists `saleYear`/`saleMonth` and `comparableCount`.
   - Optional derivations via YAML: street×year medians (`streetYearMedianPrice`), postcode prefix, and `priceFactor = salePrice / streetYearMedianPrice`.
   - Median‑imputes numerics, fills categorical nulls, buckets sparse categories; writes `data_preprocess/cleaned.parquet` and `metadata.json`.
2. Suburb Medians (`src/suburb_median.py`)
   - Aggregates suburb×month medians with counts into `data_training/suburb_month_medians.parquet`.
   - Trains a Gradient Boosting forecaster using time index + seasonality and suburb one‑hots; saves `models/suburb_median_model.pkl` and `suburb_median_model_meta.json`.
   - Provides forecasted medians for missing suburb/month pairs with a global fallback.
3. Feature Selection (`src/feature_selection.py`)
   - Loads cleaned data, removes identifiers and time columns, applies exclusions from `config/feature_engineering.yml`.
   - Joins suburb medians to compute a factor target when configured, prunes low‑variance and highly‑correlated numerics, ranks with a decision tree.
   - Guarantees locality features (`street`, `suburb`, `propertyType`, `bed`, `bath`, `car`, `comparableCount`, `saleYear`, `saleMonth`) remain when present.
   - Persists `X.parquet`, `y.parquet`, train/val splits, `feature_metadata.json`, and `feature_importances.json`.
4. Model Training (`src/model_training.py`)
   - Reads `config/model.yml` for split, manual feature adjustments, and enabled regressors/grids.
   - Supports `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`, plus `Lasso`/`ElasticNet` (via `MODEL_FACTORY`).
   - If target is a factor, translates predictions back to price using suburb baselines before scoring; logs MAE/RMSE/R² and saves `models/best_model.pkl` with `best_model.json` and `model_metrics.json`.

## Streamlit Application
Launch after artefacts exist:
```bash
streamlit run app/app.py
```
- Overview: suburb filter, year range, basic street‑level summaries. Map/heatmap appears once `config/street_coordinates.csv` is provided (columns: `suburb`, `street`, `latitude`, `longitude`; streets title‑cased, suburbs upper‑case).
- Pages:
  - Data Preprocessing: run the pipeline and explore raw/cleaned/derived datasets with metadata.
  - Feature Engineering: explore correlations, view/importances, and manage selections (interactive helpers).
  - Model Selection: edit YAML, toggle models, grid‑search, and view persisted metrics/selection.

Note: A dedicated “Predict” form is not included yet in `app/pages/`. Predictions can be added later using the saved `best_model.pkl` and suburb median utilities.

## Generated Artefacts
- `data_preprocess/cleaned.parquet` and `metadata.json` (includes raw file signatures to detect change).
- `data_training/suburb_month_medians.parquet` plus `models/suburb_median_model.pkl` and `suburb_median_model_meta.json`.
- `data_training/X.parquet`, `y.parquet`, `X_train.parquet`, `X_val.parquet`, `y_train.parquet`, `y_val.parquet`, `feature_metadata.json`, `feature_importances.json`.
- `models/best_model.pkl`, `best_model.json`, `model_metrics.json`.

## Configuration
- `config/preprocessing.yml` – data source, include list, category mappings, and derivations (street extraction, street×year medians, postcode prefix, price factor, required columns).
- `config/feature_engineering.yml` – target type, baseline handling, correlation threshold, drop/force‑keep lists.
- `config/model.yml` – split, enabled models, hyper‑parameter grids, and manual feature include/exclude.

## Maintenance
- Add/replace CSVs in `data/` and rerun: `python -m src.run_pipeline --force`.
- Ensure `models/best_model.pkl` matches latest metadata before using the app.
- Keep the virtual environment up‑to‑date (scikit‑learn, streamlit).

## Troubleshooting
- Missing artefacts: rerun the pipeline; `--force` if raw files changed outside the signature logic.
- Streamlit errors: activate the venv and ensure artefacts are present.
- Step-by-step: run individual modules via `python -m src.preprocess`, `python -m src.suburb_median`, `python -m src.feature_selection`, `python -m src.model_training` to inspect stages.
