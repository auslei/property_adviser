## Property Analysis Agents Guide

### Overview
- End-to-end pipeline for forecasting property prices using suburb-level data.
- Core stages are modularised:
  0. Macro data ingestion (CPI, cash rate, ASX) — optional enrichment
  1. Preprocessing (cleaning + derivations)
  2. Feature selection (scoring, guardrails, overrides)
  3. Model training / selection
  4. Prediction on persisted artefacts
- A Streamlit UI wraps the pipeline for interactive runs.

### Reference Docs
- MACRO.md — macroeconomic fetcher and integration helper
- PREPROCESS_MODULE.md — preprocessing pipeline breakdown
- FEATURE_SELECTION.md — feature scoring, guardrails, and overrides
- TRAINING.md — model training workflow and artefacts
- COMMON.md — shared conventions, data contracts, glossary
- DEV_GUIDELINES.md — coding standards and agent workflows

---

## Pipeline Stages

### 0) Macro Data (`property_adviser/macro`)
- CLI: `uv run pa-macro --config config/macro.yml --verbose`
- Fetches RBA CPI (quarterly + annual), RBA cash rate, and ASX index series, writing CSVs to `data/macro/` plus a merged `macro_au_annual.csv`.
- `add_macro_yearly(df, macro_path=...)` joins macro features onto derived property data by sale year (see MACRO.md for column schema).

### 1) Preprocessing (`property_adviser/preprocess`)
- `preprocess/cli.py` orchestrates cleaning and derivation via `config/preprocessing.yml`.
- Cleaning (`preprocess_clean.py`): renames columns, fixes categorical noise, coerces numerics, audits dropped rows if `dropped_rows_path` is set.
- Derivation (`preprocess_derive.py`): seasonality encodings, suburb rolling stats, ratio features, age bands, optional macro joins.
- Outputs land in `data/preprocess/`: `cleaned.csv`, `derived.csv`, `metadata.json`, optional `dropped_rows` parquet.

### 2) Feature Selection (`property_adviser/feature`)
- Inputs: `derived.csv` and `config/features.yml` (target, threshold/top-k, guardrail settings).
- Metrics: `pearson_abs`, **normalised** `mutual_info`, `eta`; `best_score` drives selection.
- Guardrails: drop ID-like columns, enforce family preferences, prune correlated pairs (all annotated in the `reason` column).
- Manual overrides stay GUI-friendly: `include`, `exclude`, `use_top_k`, `top_k`.
- Outputs in `data/training/`: `feature_scores.parquet`, `X.csv`, `y.csv`, `training.csv`, `selected_features.txt`.
- Programmatic entrypoint `run_feature_selection` mirrors CLI behaviour for Streamlit usage.

### 3) Model Training / Selection (`property_adviser/train`)
- CLI: `uv run pa-train --config config/model.yml --verbose`.
- Respects manual selections in the scores table, performs month-based train/validation split (auto fallback to latest month), and wraps models in a preprocessing pipeline.
- Supports `log_target: true` for log-transform training, plus per-model GridSearch definitions (defaults cover Linear/Ridge/Lasso/ElasticNet/RF/GBR).
- Artefacts saved to `models/`: timestamped `best_model_*.joblib`, `model_scores_*.csv`, with metadata enumerating validation month and feature list.

### 4) Prediction (`property_adviser/predict`)
- Loads a persisted pipeline (`best_model_*.joblib`) and produces scored outputs for new records.

---

## Streamlit Application
- `app/Overview.py` — high-level dashboard
- `app/pages/1_Data_Preprocessing.py` — run/inspect preprocessing outputs and metadata
- `app/pages/2_Feature_Engineering.py` — review scores, apply overrides, regenerate `X`/`y`
- `app/pages/3_Model_Selection.py` — kick off training, inspect validation metrics, download artefacts
- `app/pages/4_Model_Prediction.py` — interactively score new data with the selected model

---

## Configuration
- Macro: `config/macro.yml`
- Preprocess: `config/preprocessing.yml` → `pp_clean.yml`, `pp_derive.yml`
- Features: `config/features.yml`
- Model: `config/model.yml`
- Optional reference data: `config/street_coordinates.csv`
- Shared contracts: `docs/COMMON.md`

---

## Development Guideline (summary)
- Stages stay decoupled; shared utils live under `property_adviser/core`.
- Streamlit triggers module functions; keep heavy logic in package code for reuse by agents.
- Config-driven design keeps runs reproducible across agents; prefer YAML edits over code forks.
- Follow `docs/DEV_GUIDELINES.md` for coding standards, review expectations, and agent hand-offs.
