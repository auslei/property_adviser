# Property Analyst

## Overview
Property Analyst builds an end-to-end workflow for forecasting next-year property sale prices from suburb-level records. The pipeline ingests raw CSV exports, standardises and enriches the data, selects the most informative attributes, trains several regression models, and exposes the best-performing model through a Streamlit UI for interactive predictions.

## Repository Layout
- `data/` – drop raw suburb-level CSV exports here (input files).
- `data_preprocess/` – cleaned dataset (`cleaned.parquet`) and preprocessing metadata (`metadata.json`).
- `data_training/` – feature matrix/target parquet files plus feature metadata.
- `models/` – fitted model artefacts and evaluation metrics.
- `config/` – YAML configurations for preprocessing, feature engineering, and model selection.
- `src/` – Python modules for preprocessing, feature selection, model training, and pipeline orchestration.
- `app/` – Streamlit overview page and shared utilities.
- `app/pages/` – multi-page Streamlit workflows (preprocessing, feature engineering, model selection).

## Requirements
- Python 3.10 or newer (tested with CPython).
- Local CSV data files located in `./data`.

## Quick Start
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your latest property CSV exports in the `data/` directory. Multiple files are supported; they will be concatenated during preprocessing.

## Running the Pipeline
Run the full data-preparation and modelling workflow with:
```bash
python -m src.run_pipeline
```
Use `--force` to regenerate the preprocessed dataset even if the raw files are unchanged:
```bash
python -m src.run_pipeline --force
```

### Pipeline Components
1. **Preprocessing (`src/preprocess.py`)**
   - Loads raw CSVs based on `config/preprocessing.yml` (path, pattern, and column include list).
   - Normalises column names, drops mostly empty columns, applies category mappings, and derives configurable features (e.g. street extraction, street/year aggregates, postcode prefixes).
   - Reformats `saleDate` to `YYYYMM`, persists `saleYear`/`saleMonth`, enforces required fields, computes `comparableCount`, and writes `data_preprocess/cleaned.parquet` plus `metadata.json`.
2. **Suburb Median Baselines (`src/suburb_median.py`)**
   - Aggregates suburb × year-month medians and transaction counts from the cleaned dataset into `data_training/suburb_month_medians.parquet`.
   - Fits a Gradient Boosting forecaster over the historical medians with time/seasonality features and suburb one-hot encodings, persisting the model at `models/suburb_median_model.pkl` with metadata.
   - Supplies forecasted suburb medians when observations are missing, backing off to a global baseline where needed.
3. **Feature Selection (`src/feature_selection.py`)**
   - Guided by `config/feature_engineering.yml` (target choice, baseline options, correlation threshold, drop/keep lists).
   - Removes identifiers, attaches suburb medians, derives factor targets if configured, prunes low-variance and highly correlated features, and ranks candidates with a decision tree regressor.
   - Persists `data_training/X.parquet`, `y.parquet`, train/val splits, feature importances, and metadata describing the configuration used.
4. **Model Training (`src/model_training.py`)**
   - Reads `config/model.yml` to control train/validation split, manual feature adjustments, and enabled regressors/hyper-parameter grids.
   - Supports linear models, random forest, gradient boosting, Lasso, and ElasticNet out of the box (extendable via `MODEL_FACTORY`).
   - Evaluates price-factor or direct price targets, logs MAE/RMSE/R² metrics, and saves the best pipeline to `models/best_model.pkl` with selection metadata.
5. **Streamlit Application (`app/app.py` & `app/pages/`)**
   - Multi-page experience for configuring preprocessing, feature engineering, model training, and exploring the overview heatmap/prediction UI.

## Using the Streamlit App
After the pipeline completes and the best model is saved, start the UI:
```bash
streamlit run app/app.py
```
- **Overview**: interactive street heatmap layered on a map, correlation-driven filters, and latest model metrics. Requires `config/street_coordinates.csv` containing `suburb`, `street`, `latitude`, `longitude` (streets title-cased, suburbs upper-case, coordinates in decimal degrees).
- **Data Preprocessing**: edit `config/preprocessing.yml`, preview raw inputs based on the configured path/pattern, and run the cleaning pipeline in-app.
- **Feature Engineering**: tweak `config/feature_engineering.yml`, adjust correlation thresholds/targets/drop lists, and regenerate feature matrices.
- **Model Selection**: modify `config/model.yml`, toggle candidate regressors, customise hyper-parameters, and trigger training with persisted best-model metadata.
- **Predict**: once artefacts exist, supply property attributes to estimate next-year sale prices.

## Updating with New Data
1. Add the new CSV exports to `data/` (replace or append).
2. Rerun the pipeline (`python -m src.run_pipeline --force`).
3. Restart the Streamlit app to load the updated model artefacts.

## Troubleshooting
- **Missing files**: If `data_preprocess/cleaned.parquet` or other artefacts are absent, rerun the pipeline.
- **Streamlit errors**: Ensure the virtual environment is active and that the pipeline ran successfully before launching the app.
- **Custom workflows**: Individual steps can be run via `python -m src.preprocess`, `python -m src.feature_selection`, and `python -m src.model_training` if you need to inspect intermediate outputs.

## Configuration files
- `config/preprocessing.yml` – raw data source, column include list, category mappings, and derived feature specifications (street extraction, street/year aggregates, postcode prefixes, required columns).
- `config/feature_engineering.yml` – target selection, baseline handling, correlation threshold, drop/force-keep lists, and factor derivation options.
- `config/model.yml` – train/validation split, enabled regressors, hyper-parameter grids, and manual feature include/exclude overrides for training.
