# Property Analysis Agents Guide

## Overview
- End-to-end pipeline for forecasting next-year property prices using suburb-level CSV data. 
- Core stages live under `src/`: preprocessing, feature selection, model selection and training, and model prediction with persisted parquet artefacts.
- Streamlit UI (`app/app.py`): offering access to the preprocessing, feature selection, model selection and training, and model prediction

## Development Guideline:
- Each of the 1. preprocessing, 2. feature selection, 3. model selection and 4. model prediction should be able to run independently. 
- Common functions should be separated by functionality and stored in modules under src/common
- Source code should not be too large, split the code into functional components if required. (For example, preprocessing is split into clean, derive, and preprocessing main)
- In streamlit app:
   - All calculations that are not part of visualisation should be part of function in each of the 4 modules.
- Ensure code is concise and simple to understand, do not overengineer.
- If configration is required, expect to be in the config file, do not create inline default values.


## Environment Setup
- Create a virtual environment (e.g. `python -m venv .venv && source .venv/bin/activate`).
- Install dependencies from `requirements.txt` (`pip install -r requirements.txt`).
- Project expects local CSV files in `data/`; no remote downloads occur.

## Functionality:
1. Preprocess Mododule:
   - objectives:
      - cleaning: read raw file, clean column names, remove special characters, and drop mostly null columns
      - derivation: create new columns based on derivation logic, such as bucket categories, calculate mean values, etc.
   - location: under src, and name start with preprocess
   - config: conifig/prprocessing.yml
2. Feature selection:
   - objective:
      - given a target varible, find most useful features for modelling
      - store selected features under "data_training" directory 
      - generate training, datasets based on selected variables under "data_training" directory
   - location: under src, and name starts with feature_selection
   - config: config/features
3. Model training and selection:
   - 

- Streamlit UI (`app/app.py`) wraps the trained artefacts to offer interactive predictions and dataset exploration.
   0. app.py shows an overview of the project and data:
      - This include several filters: Suburb, saleYear, propertyType, Street Contains
      - A highest median priced street. 
      - A data table showing filtered data.
   1. pages/1_Data_Preprocessing.py: a data processing page, allowing user to:
      - regenerate all data files 
      - filter and profile data, there are two tabs
         - tab1 (general): data table showing filtered result
         - tab2 (profile): showing min, max, median, average of numerical columns; frequency count of the categorical columns
   2. 2_Feature_Engineering.py: feature selection screen:
      - allow user to specify a target variable from list of available columns (numerical)
      - shows the list of variables (both numerical and categorical) ranked by their correlation to the target variable
      - a list of features that are automatically selected, populated in selected features 
      - 2 buttons:
         - auto

 


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
