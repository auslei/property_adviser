# Training Module Agent Guide

## Purpose & Scope
- Train, select, and persist regression models using the curated features and target produced by the feature-selection stage.
- Provide a consistent CLI (`pa-train`) and programmatic API for automation and notebooks.
- Share preprocessing pipelines, logging, and IO helpers via `property_adviser.core` to ensure uniform behaviour across modules.

## Design Commitments
- **Clear interface**: Inputs are the feature matrix (`X`), target (`y`), and optional feature scores referenced via `config/model.yml`; outputs are timestamped model bundles and score summaries under `models/` and `data/training/`.
- **High cohesion**: Model selection, evaluation, and persistence live entirely within this package. External callers interact only through documented functions.
- **Low coupling**: Downstream prediction loads persisted artefacts without importing training internals. Upstream stages influence behaviour through configs rather than code paths.
- **Reusability**: Shared preprocessors, imputation defaults, and logging frameworks are built once and reused via `core`.

## Structure
```
property_adviser/train/
  config.py     # Typed configuration schema + loader
  pipeline.py   # run_training orchestration + persistence
  cli.py        # CLI entry point (pa-train)
  model_training.py  # Compatibility wrapper around the pipeline
```

## Configuration (`config/model.yml`)
- `task`, `target`, `log_target`
- `input`: base path plus filenames for `X`, `y`, and optional `feature_scores`
- `model_path.base`: directory for persisted artefacts
- `split`: validation month logic (`validation_month`, `month_column`)
- `models`: enabled estimators and GridSearch parameter grids

Keep configuration declarative; new behaviour should be toggled via YAML rather than code edits.

## Workflow Summary
1. Load `TrainingConfig` via `load_training_config`, which resolves paths relative to the repo root (`PROJECT_ROOT`) and honours overrides.
2. Read X/y (and optional feature scores) via `core.io.load_parquet_or_csv`, applying manual selections before modelling.
3. Choose the validation month (requested or latest) and perform a month-based split, dropping the month column from model inputs.
4. Build a single preprocessing pipeline (numeric: median + scaler, categorical: mode + one-hot) reused across every estimator.
5. For each enabled model, run `GridSearchCV` (default R² scoring, 3-fold) inside a timed block; validation metrics and CV scores are logged per candidate.
6. Persist the winning model bundle, score summaries, and feature metadata (including imputation defaults) with timestamped filenames.
7. Emit `train.complete` with total runtime, best model, and validation month; return a `TrainingResult` dataclass for downstream use.

## Outputs
- `models/best_model_<name>_<timestamp>.joblib`
- `models/best_model.joblib` (canonical symlink/copy) and `models/best_model.json`
- `models/model_scores_<timestamp>.csv`
- `data/training/feature_metadata.json`

All writes use `core.io.ensure_dir` + `joblib.dump`/`json.dump` wrappers for consistency.

## CLI Usage
```bash
uv run pa-train --config config/model.yml --verbose
```
- Optional overrides (`--model`, `--log-target`, etc.) should map directly to config keys.

## Programmatic Usage
```python
from pathlib import Path
from property_adviser.train import load_training_config, run_training

config = load_training_config(Path("config/model.yml"))
result = run_training(config)
print(result.best_model, result.validation_month)
```

## Handover to Prediction
- Ensure `feature_metadata.json` stays in sync with the persisted pipeline.
- Document any changes to bundle structure here and in `property_adviser/predict/AGENTS.md`.

## Maintenance Checklist
1. Keep preprocessing logic centralised; if additional transformers benefit multiple models, extract them into helpers.
2. Extend estimator support by updating `MODEL_FACTORY` and documenting new configuration flags.
3. Backward compatibility matters: when changing artefact names or metadata schemas, update dependent modules and docs.
