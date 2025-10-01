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
  config.py     # Typed configuration schema + loader (multi-target aware)
  pipeline.py   # run_training orchestration + persistence
  cli.py        # CLI entry point (pa-train)
  model_training.py  # Compatibility wrapper around the pipeline
```

## Configuration (`config/model.yml`)
- Declare shared defaults once (`models`, `model_path.base`, `task`, `log_target` etc.).
- List training targets under `targets`, each with its own `name`, `target` column, `input` paths, optional `split` overrides, and per-target model tweaks.
- Artefacts are written to `<model_path.base>/<target>/` unless overridden.

Keep configuration declarative; extend or override behaviour via YAML rather than code edits.

## Workflow Summary
1. Load one or more `TrainingConfig` instances via `load_training_config`; each represents a distinct target/horizon.
2. Read X/y (and optional feature scores) via `core.io.load_parquet_or_csv`, applying manual selections before modelling.
3. Choose the validation month (requested or latest), sort chronologically, and perform a month-based split, dropping the month column from model inputs.
4. Build a single preprocessing pipeline (numeric: median + scaler, categorical: mode + one-hot) reused across every estimator.
5. For each enabled model, run `GridSearchCV` with a `TimeSeriesSplit` (default up to 5 folds based on sample count, R² scoring) inside a timed block; validation metrics and CV scores are logged per candidate and per target.
6. Persist the winning model bundle, score summaries, and feature metadata (including imputation defaults) with timestamped filenames under the target’s artefact directory.
7. Emit `train.complete` with total runtime, best model, and validation month; return a `TrainingResult` dataclass. `train_timeseries_model` aggregates these results and emits a holistic JSON report.

## Outputs
- `models/<target>/best_model_<estimator>_<timestamp>.joblib`
- `models/<target>/best_model.joblib` and `models/<target>/best_model.json`
- `models/<target>/model_scores_<timestamp>.csv`
- `data/training/<target>/feature_metadata.json`
- `models/training_report_<timestamp>.json` – consolidated summary across all targets.

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

configs = load_training_config(Path("config/model.yml"))
for cfg in configs:
    result = run_training(cfg)
    print(cfg.name, result.best_model, result.validation_month)
```

## Handover to Prediction
- Ensure `feature_metadata.json` stays in sync with the persisted pipeline.
- Document any changes to bundle structure here and in `property_adviser/predict/AGENTS.md`.

## Maintenance Checklist
1. Keep preprocessing logic centralised; if additional transformers benefit multiple models, extract them into helpers.
2. Extend estimator support by updating `MODEL_FACTORY` and documenting new configuration flags.
3. Backward compatibility matters: when changing artefact names or metadata schemas, update dependent modules and docs.
