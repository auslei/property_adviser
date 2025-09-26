## Property Analysis Agents Guide

### Overview
- End-to-end pipeline for forecasting property prices using suburb-level data.  
- Core stages are modularised:
  1. Preprocessing (cleaning + derivations)
  2. Feature selection
  3. Model training/selection
  4. Prediction (using persisted parquet/model artefacts)
- A Streamlit UI wraps the pipeline for interactive runs.

### Reference Docs
- PREPROCESS_MODULE.md — preprocessing pipeline breakdown
- FEATURE_SELECTION.md — feature scoring and selection details
- TRAINING.md — model training workflow and artefacts
- COMMON.md — shared conventions, data contracts, glossary
- DEV_GUIDELINES.md — coding standards and agent workflows

---

## Pipeline Stages

### 1) Preprocessing (`property_adviser/preprocess`)
- **Cleaning** (`preprocess_clean.py`): standardises categories and consolidates messy text. (See PREPROCESS_MODULE.md.)
- **Derivations** (`preprocess_derive.py`): computes features (ratios, per-area prices, medians, cyclical encodings, etc.).
- **CLI** (`cli.py`): orchestrates cleaning + derivation via split configs.
- Config split:
  - `config/pp_clean.yml` (cleaning)
  - `config/pp_derive.yml` (derivations)
  - `config/preprocessing.yml` (orchestrator)

### 2) Feature Selection (`property_adviser/feature`)
- Inputs: **derived dataset** (from step 1) and `config/features.yml`. Full behaviour in FEATURE_SELECTION.md.
- Methods:
  - Filter/embedded metrics per feature: `pearson_abs` (|corr|), **normalised** `mutual_info` [0–1], and `eta` for categorical.
  - Global threshold on the **best_score** across these metrics.
  - Optional **top-k** selection mode (rank by best_score).
- **Manual overrides (GUI-friendly):**
  - `include`: always selected (reason = “manual include”)
  - `exclude`: never selected (reason = “manual exclude (not selected)”)
  - `use_top_k`: None → follow config; True/False → force mode
  - `top_k`: override value (if enabled)
- **Outputs:**
  - `X.parquet`, `y.parquet`
  - `training.parquet` (X + y combined, optional compatibility)
  - Single **scores + selection** file (default `feature_scores.parquet`) containing:
    - `feature, pearson_abs, mutual_info, eta, best_metric, best_score, selected, reason`
- **Entrypoints**
  - **CLI**: `uv run pa-feature --config config/features.yml --scores-file feature_scores.parquet`
  - **Code/GUI**: `run_feature_selection(cfg, include=[], exclude=[], use_top_k=None, top_k=None, write_outputs=False)`

### 3) Model Training/Selection (`property_adviser/train`)
- Trains multiple regressors, evaluates metrics (MAE/RMSE/R²), and saves the best model. Refer to TRAINING.md for configuration nuances and logging events.

### 4) Prediction (`property_adviser/predict`)
- Loads the best model and scores new inputs.

---

## Streamlit Application
- `app/Overview.py` — high-level dashboard
- `app/pages/1_Data_Preprocessing.py` — run/inspect preprocessing
- `app/pages/2_Feature_Engineering.py` — choose target, view metrics, apply overrides (`include/exclude`, top-k), generate X/y
- `app/pages/3_Model_Selection.py` — training and metrics (see TRAINING.md for backend behaviour)
- `app/pages/4_Model_Prediction.py` — interactive scoring

---

## Configuration
- Preprocess: `config/preprocessing.yml` → points to `pp_clean.yml`, `pp_derive.yml`
- Features: `config/features.yml`
- Model: `config/model.yml`  
- Optional: `config/street_coordinates.csv`
- Shared conventions and schema expectations: COMMON.md

---

## Development Guideline (summary)
- Each stage runs independently; shared utilities live under `property_adviser/core`.
- Streamlit pages orchestrate; heavy logic stays in modules.
- All behaviour is YAML-driven for reproducibility.
- Follow DEV_GUIDELINES.md for coding standards, review expectations, and agent hand-offs.
