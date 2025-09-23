# Property Analysis Agents Guide

## Overview
- End-to-end pipeline for forecasting next-year property prices using suburb-level CSV data.
- Core stages live under `src/`: preprocessing, feature selection, and model training with persisted parquet artefacts.
- Streamlit UI (`app/app.py`) wraps the trained artefacts to offer interactive predictions and dataset exploration.

## Environment Setup
- Create a virtual environment (e.g. `python -m venv .venv && source .venv/bin/activate`).
- Install dependencies from `requirements.txt` (`pip install -r requirements.txt`).
- Project expects local CSV files in `data/`; no remote downloads occur.

## Data Inputs
- Place raw CSV files in `data/` (matched via `*.csv`). UTF-8 with BOM is accepted.
- Pipeline records file size and modified timestamps to detect changes; metadata is stored alongside preprocessed outputs.

## Pipeline Stages
1. **Preprocessing** (`src/preprocess.py`)
   - Normalises column names to camelCase, removing symbols and duplicates.
   - Drops columns with <35% non-null values and strips/standardises string fields.
   - Extracts `street` from `streetAddress`, simplifies `propertyType`, coerces numeric candidates, and reformats `saleDate` to `YYYYMM`.
   - Persists `saleYear`/`saleMonth` integers and removes rows lacking `salePrice`, `bed`, or `bath`; computes `comparableCount` for repeat street/type/bed/bath/car pairs.
   - Imputes remaining numerics with the median, fills categorical nulls with `Unknown`, buckets sparse categories, then saves `data_preprocess/cleaned.parquet` with `metadata.json`.
2. **Suburb Median Baselines** (`src/suburb_median.py`)
   - Aggregates suburb × year-month medians (with transaction counts) from the cleaned dataset and stores them at `data_training/suburb_month_medians.parquet`.
   - Trains a Gradient Boosting forecaster that models medians using time index/seasonality with suburb one-hot features, persisting the model (`models/suburb_median_model.pkl`) and metadata (`suburb_median_model_meta.json`).
   - Provides forecasted medians when a suburb/month pair lacks observations, falling back to a global baseline if necessary.
3. **Feature Selection** (`src/feature_selection.py`)
   - Loads the cleaned parquet, removes identifiers/time columns, and applies exclusions defined in `config/feature_engineering.yml`.
   - Joins suburb medians to compute a `priceFactor = salePrice / baselineMedian`, dropping invalid rows and retaining `saleYear`/`saleMonth` for inference.
   - Drops low-variance fields (while preserving time keys), prunes highly correlated numerics, then fits a decision tree regressor to rank features.
   - Guarantees key locality features (`street`, `suburb`, `propertyType`, `bed`, `bath`, `car`, `comparableCount`, `saleYear`, `saleMonth`) remain when available and persists factor targets at `data_training/X.parquet` & `y.parquet` plus metadata/importance files.
4. **Model Training** (`src/model_training.py`)
   - Trains on the saved matrices with an 80/20 split against the factor target.
   - Evaluates `LinearRegression`, `RandomForestRegressor`, and `GradientBoostingRegressor` (with GridSearchCV for tree-based models), translating factor predictions back to price using the suburb baseline before scoring.
   - Logs MAE/RMSE/R² (price) alongside factor metrics, saves the best pipeline to `models/best_model.pkl`, and writes selection metadata (`best_model.json`, `model_metrics.json`).

## Running the Pipeline
- Execute end-to-end via `python -m src.run_pipeline` (add `--force` to re-run preprocessing even if raw files are unchanged).
- Intermediate parquet and JSON artefacts are overwritten each run; downstream steps consume these outputs.

## Generated Artefacts
- `data_preprocess/cleaned.parquet` + `metadata.json`: canonical cleaned dataset with source signatures.
- `data_training/suburb_month_medians.parquet` (baseline history) plus `models/suburb_median_model.pkl`/`suburb_median_model_meta.json` for forecasting missing medians.
- `data_training/X.parquet`, `y.parquet`, `X_train.parquet`, `X_val.parquet`, `y_train.parquet`, `y_val.parquet`, `feature_metadata.json`, `feature_importances.json`.
- `models/best_model.pkl`, `best_model.json`, `model_metrics.json` for deployment and diagnostics.

## Streamlit Application
- Launch with `streamlit run app/app.py` after the pipeline populates artefacts.
- Tabs:
  - **Predict**: form-driven categorical/numeric inputs; selects sale year/month, forecasts or retrieves the suburb median, then multiplies it by the predicted factor (with comparable-count safeguards).
  - **Explore Data**: inspect raw/preprocessed/training datasets, view suburb-month baseline tables, and (when `config/street_coordinates.csv` is supplied) interact with the Mitcham street heatmap.
  - **Feature Insights**: visualises feature importances, model metrics, and best-model parameters.

## Configuration & Customisation
- Optional adjustments can be made via the YAML files in `config/` (preprocessing, feature_engineering, model).
- Provide street coordinates for the heatmap via `config/street_coordinates.csv` (columns: `suburb`, `street`, `latitude`, `longitude`; streets should be title-cased).
- Adjust thresholds (e.g. minimum non-null fraction, category bucketing) directly in `src/config.py` or preprocessing helpers if model requirements change.

## Maintenance Checklist
- When new CSVs are added to `data/`, re-run the pipeline to refresh artefacts and retrain models.
- Confirm `models/best_model.pkl` aligns with the latest metadata before deploying or starting the Streamlit app.
- Keep the virtual environment updated with evolving dependency needs (especially scikit-learn and streamlit versions).
