# Property Analyst

## Overview
Property Analyst builds an end-to-end workflow for forecasting next-year property sale prices from suburb-level records. The pipeline ingests raw CSV exports, standardises and enriches the data, selects the most informative attributes, trains several regression models, and exposes the best-performing model through a Streamlit UI for interactive predictions.

## Repository Layout
- `data/` – drop raw suburb-level CSV exports here (input files).
- `data_preprocess/` – cleaned dataset (`cleaned.parquet`) and preprocessing metadata (`metadata.json`).
- `data_training/` – feature matrix/target parquet files plus feature metadata.
- `models/` – fitted model artefacts and evaluation metrics.
- `src/` – Python modules for preprocessing, feature selection, model training, and pipeline orchestration.
- `app/` – Streamlit interface for price prediction.

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

### What Happens
1. **Preprocessing (`src/preprocess.py`)**
   - Cleans column names (camelCase with safe ASCII characters).
   - Drops columns with <35% non-null values.
   - Extracts `street` details from `streetAddress`.
   - Converts date columns to `YYYYMM` plus `saleYear`/`saleMonth`.
   - Imputes numeric fields with the median and fills categorical fields with `Unknown`.
   - Buckets high-cardinality categoricals into user-friendly groups.
2. **Feature Selection (`src/feature_selection.py`)**
   - Removes near-constant fields and strongly correlated numeric features.
   - Fits a decision tree to score feature importance.
   - Persists the selected feature set, metadata, and train/target parquet files.
3. **Model Training (`src/model_training.py`)**
   - Splits into train/validation folds.
   - Evaluates Linear Regression, Random Forest, and Gradient Boosting models (with hyper-parameter search where appropriate).
   - Saves metrics (`models/model_metrics.json`) and the best pipeline (`models/best_model.pkl`).

## Using the Streamlit App
After the pipeline completes and the best model is saved, start the UI:
```bash
streamlit run app/app.py
```
- Choose categorical options and numeric ranges derived from the training data.
- Click **Predict Price** to view the estimated sale price for the upcoming year.
- The sidebar displays the best model name and validation R².

## Updating with New Data
1. Add the new CSV exports to `data/` (replace or append).
2. Rerun the pipeline (`python -m src.run_pipeline --force`).
3. Restart the Streamlit app to load the updated model artefacts.

## Troubleshooting
- **Missing files**: If `data_preprocess/cleaned.parquet` or other artefacts are absent, rerun the pipeline.
- **Streamlit errors**: Ensure the virtual environment is active and that the pipeline ran successfully before launching the app.
- **Custom workflows**: Individual steps can be run via `python -m src.preprocess`, `python -m src.feature_selection`, and `python -m src.model_training` if you need to inspect intermediate outputs.
# property_adviser
