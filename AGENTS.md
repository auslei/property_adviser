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

1. **Geocoding (`property_adviser/geocode`)**
   - CLI: `uv run pa-geocode --config config/geocode.yml --verbose`.
   - Enriches the cleaned dataset with latitude and longitude information.
   - The level of geocoding (street or full address) is configurable.
   - Output is saved to `data/geocode/geocoded.parquet`.
   - See `property_adviser/geocode/AGENTS.md` for more details.

2. **Cleaning (`property_adviser/clean`)**
   - CLI: `uv run pa-clean --config config/clean.yml --verbose`.
   - Cleans the raw data and saves it to `data/clean/cleaned.parquet`.
   - See `property_adviser/clean/AGENTS.md` for more details.

3. **Derivation (`property_adviser/derive`)**
   - CLI: `uv run pa-derive --config config/derive.yml --verbose`.
   - Takes the cleaned data, joins it with macro and geocoded data, and derives features.
   - Saves the final dataset to `data/derive/derived.parquet`.
   - See `property_adviser/derive/AGENTS.md` for more details.

3. **Feature Selection (`property_adviser/feature`)**
   - Typed config loader + pipeline record elapsed timings, normalise scores, and expose consistent guardrail logging.
   - Supports correlation threshold or top-k selection, optional RFECV pruning with row/feature caps, and emits `feature_scores` plus X/y artefacts per target (`data/features/<target>/`). Existing files are overwritten.
   - CLI iterates through all targets in `config/features.yml`: `uv run pa-feature --config config/features.yml --scores-file feature_scores.parquet`.
   - Implementation notes, elimination tuning, and API usage: `property_adviser/feature/AGENTS.md`.

4. **Model Training (`property_adviser/train`)**
   - Typed configs (`load_training_config`) feed `run_training`, which logs per-stage timings and produces `TrainingResult` objects.
   - Performs month-based train/validation split, applies manual feature overrides, and supports shared preprocessing pipelines per target.
   - Persists bundles under `models/<YYYYMMDD>/<target>/` with stable filenames (`best_model.joblib`, `best_model.json`, `model_scores.csv`) so same-day reruns overwrite in place. Emits a daily `training_report.json` under `models/<YYYYMMDD>/`.
   - Detailed workflow and extension tips in `property_adviser/train/AGENTS.md`.

5. **Prediction (`property_adviser/predict`)**
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
- Geocode: `config/geocode.yml`
- Clean: `config/clean.yml`
- Derive: `config/derive.yml`
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

### Recent Updates (2025-09-30 21:35:00Z)
- Preprocessing now produces configurable property buckets, segment-level aggregates, and forward targets saved to `segments.parquet` (with detailed row-level snapshots retained separately).
- Feature selection and model training process multiple targets/horizons per run, writing outputs to per-target directories and logging durations for each stage.
- Training orchestrator emits a consolidated JSON report summarising best models and metrics across all configured targets for downstream monitoring.
