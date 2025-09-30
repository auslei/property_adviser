## Property Analysis Agents Guide

### Design Principles
- Ship each module as a cohesive unit with a clear public interface (CLI + Python entrypoints) and minimal cross-module coupling.
- Concentrate reusable helpers, configuration loaders, and IO tools inside `property_adviser/core` and import them instead of rewriting logic downstream.
- Maintain consistency across stages: common logging, config loading, and dataset contracts should feel identical whether invoked via CLI or notebooks.
- Document ownership close to the code: every package exposes an `AGENTS.md` describing its interface, dependencies, and hand-offs.

### Pipeline Overview
0. **Macro Data (`property_adviser/macro`)**
   - CLI: `uv run pa-macro --config config/macro.yml --verbose`.
   - Delivers CPI, cash-rate, and ASX series into `data/macro/` plus `macro_au_annual.csv` for downstream joins.
   - Public helper `add_macro_yearly(df, macro_path=...)` merges annual macro fields by sale year.
   - See `property_adviser/macro/AGENTS.md` for source contracts and configuration patterns.

1. **Preprocessing (`property_adviser/preprocess`)**
   - `preprocess/cli.py` orchestrates cleaning and derivation using `config/preprocessing.yml`.
   - Cleaning normalises schema and dtype noise while derivation adds seasonality, suburb rolling stats, ratios, age features, and optional macro enrichments.
   - Outputs land in `data/preprocess/cleaned.csv`, `derived.csv`, `metadata.json`, with optional `dropped_rows` audit.
   - Interface and dataset expectations live in `property_adviser/preprocess/AGENTS.md`.

2. **Feature Selection (`property_adviser/feature`)**
   - Consumes `derived.csv` per `config/features.yml`, scores via Pearson, normalised MI, and eta metrics, then applies guardrails and overrides.
   - Produces `feature_scores.parquet` plus `X.csv`, `y.csv`, `training.csv`, and `selected_features.txt` under `data/training/`.
   - CLI: `uv run pa-feature --config config/features.yml --scores-file feature_scores.parquet`.
   - Implementation notes and RFECV guidance: `property_adviser/feature/AGENTS.md`.

3. **Model Training (`property_adviser/train`)**
   - CLI: `uv run pa-train --config config/model.yml --verbose`.
   - Performs month-based train/validation split, wraps estimators in a shared preprocessing pipeline, respects manual feature selections, and supports log-target training.
   - Outputs timestamped bundles under `models/` plus score summaries (`model_scores_*.csv`) and feature metadata.
   - Detailed workflow in `property_adviser/train/AGENTS.md`.

4. **Prediction (`property_adviser/predict`)**
   - Loads persisted bundles (`models/best_model.joblib`) and reconstructs the feature frame expected by training.
   - Batch + single-property scoring helpers expose consistent signatures; confidence intervals leverage validation RMSE.
   - Reference `property_adviser/predict/AGENTS.md` for API usage and data contracts.

### Core Module (`property_adviser/core`)
- Centralises shared behaviours: logging, configuration loading, filesystem paths, IO helpers, and runner utilities.
- Model/pipeline artefact loaders live alongside these utilities (`core/artifacts.py`) so prediction and apps fetch bundles the same way.
- New functionality that benefits multiple stages should live here to keep modules lean and coupled only through well-defined interfaces.
- Documentation resides in `property_adviser/core/AGENTS.md`.

### Applications (`app`)
- Streamlit front ends (predictor and market insights dashboards) live in `app/` and depend on the persisted artefacts above.
- They act as presentation layers, delegating business logic back to `property_adviser` packages.
- Refer to `app/AGENTS.md` for deployment guidance and extension practices.

### Configuration Map
- Macro: `config/macro.yml`
- Preprocess: `config/preprocessing.yml` (split into `pp_clean.yml`, `pp_derive.yml`)
- Features: `config/features.yml`
- Model: `config/model.yml`
- Optional reference data: `config/street_coordinates.csv`
- Shared contracts + glossary: `docs/COMMON.md`

### Development Standards
- **Principles**: honour separation of concerns across modules, keep runs reproducible via YAML-driven configs, favour structured logging, and prefer declarative toggles over hidden defaults.
- **Configuration**: one concept per YAML file, validate schema up front, wire CLIs so `--config` always points at the controlling document.
- **Code**: author small, composable functions; place reusable helpers in `property_adviser/core`; ensure each CLI wraps a reusable function that accepts a config dict and returns typed results.
- **IO**: write/read artefacts using `core.io.save_parquet_or_csv`, `load_parquet_or_csv`, `write_list`, and `ensure_dir`; choose formats via file suffixes.
- **Logging**: emit structured events with `core.app_logging` (e.g., `feature_selection.complete`, `train.validation`) so pipelines remain traceable.
- **Testing**: cover critical cleaning and derivation steps, guardrail logic, and scoring behaviour with unit tests; maintain golden fixtures for end-to-end reproducibility.

### Recent Updates (2025-09-29 04:05:57Z)
- Feature selection now supports optional RFECV elimination, richer logging, and configurable dataset exports.
- Added `market_insights_app.py` Streamlit dashboard plus documentation so analysts can explore drivers, demand shifts, and price timelines.
- Guardrails were hardened (correlation pruning respects categoricals, driver tab honours `exclude_columns`) and docs refreshed for the new workflow.
