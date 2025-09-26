# Property Analysis Agents Guide

## Overview
- End-to-end pipeline for forecasting property prices using suburb-level data. 
- Core stages are modularised:
  1. Preprocessing (cleaning + derivations)
  2. Feature selection
  3. Model training/selection
  4. Model prediction (using persisted parquet/model artefacts)
- Streamlit UI (`app/app.py`) wraps the pipeline to offer interactive preprocessing, feature engineering, training, and prediction.

---

## Development Guideline
- Each stage (1. preprocessing, 2. feature selection, 3. model training, 4. prediction) should run independently. 
- Shared utilities live under `property_adviser/core`.
- Code should be concise and modular (e.g. preprocessing is split into `preprocess_clean.py`, `preprocess_derive.py`, `cli.py`).
- Streamlit pages only orchestrate visualisation and inputs; heavy logic belongs in modules.
- Configuration is always in YAML, not inline defaults.

👉 For full details of preprocessing design, see [PREPROCESS_MODULE.md](PREPROCESS_MODULE.md).  
👉 For coding practices, see [DEV_GUIDELINES.md](DEV_GUIDELINES.md).

---

## Environment Setup
- Create a virtual environment (or use `uv venv`).
- Sync dependencies with `uv sync` (or `pip install -r requirements.txt`).
- Project expects local CSV files in `data/`.

---

## Pipeline Stages

1. **Preprocessing** (`property_adviser/preprocess`)
   - **Cleaning** (`preprocess_clean.py`): standardises categories, consolidates messy text (e.g. `agency` → `agencyBrand`).
   - **Derivations** (`preprocess_derive.py`): computes numeric/categorical features (age, ratios, per-area prices, medians, cyclical encodings).
   - **CLI** (`cli.py`): orchestrates cleaning + derivation.
   - Config split into:
     - `config/pp_clean.yml` (cleaning rules)
     - `config/pp_derive.yml` (derivations)
     - `config/preprocessing.yml` (orchestration)

   👉 See [PREPROCESS_MODULE.md](PREPROCESS_MODULE.md) for full documentation.

2. **Feature Selection** (`property_adviser/feature`)
   - Given a target variable, ranks features and prunes unhelpful ones.
   - Persists `X`, `y`, and splits under `data_training/`.

3. **Model Training/Selection** (`property_adviser/train`)
   - Trains on persisted training data with multiple candidate models.
   - Evaluates metrics (MAE, RMSE, R²).
   - Saves best model + metadata under `models/`.

4. **Prediction** (`property_adviser/predict`)
   - Uses the persisted best model to score new inputs.
   - Consumes cleaned/derived data only (no leakage from raw).

---

## Streamlit Application
- `app/Overview.py`: overview dashboard with filters (suburb, year, type).
- `app/pages/1_Data_Preprocessing.py`: run preprocessing, filter & profile data.
- `app/pages/2_Feature_Engineering.py`: target selection, feature ranking/selection.
- `app/pages/3_Model_Selection.py`: run training/selection, inspect metrics.
- `app/pages/4_Model_Prediction.py`: interactive predictions.

---

## Configuration
- Preprocessing: `config/preprocessing.yml` → points to `pp_clean.yml`, `pp_derive.yml`
- Feature engineering: `config/features.yml`
- Model training: `config/model.yml`
- Optional: `config/street_coordinates.csv` for map views

---

## Maintenance Checklist
- When new CSVs land in `data/`, rerun preprocessing and retrain models.
- Confirm `models/best_model.pkl` aligns with the latest `model_metrics.json`.
- Keep dependencies synced with `uv`.

---

## References
- [PREPROCESS_MODULE.md](PREPROCESS_MODULE.md) — preprocess module documentation
- [DEV_GUIDELINES.md](DEV_GUIDELINES.md) — coding/development standards
