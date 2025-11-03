# Feature Selection Module Agent Guide

## Purpose & Scope
- Score and curate predictors from the derived dataset, producing training-ready feature matrices and transparent audit artefacts.
- Provide a stable CLI (`pa-feature`) and a reusable `run_feature_selection` entry point for automation.
- Reuse shared utilities from `property_adviser.core` (config loading, IO, logging) to maintain consistency across modules.

## Design Commitments
- **Clear interface**: Inputs consist of the derived dataset and `config/features.yml`; outputs are written to `data/features/` using predictable filenames (overwriting existing files).
- **High cohesion**: Scoring, guardrails, and selection logic live inside this package. When functionality becomes broadly useful (e.g., correlation helpers), promote it to `core`.
- **Low coupling**: Training and apps consume the exported files or returned dataclasses without depending on implementation details.
- **Reusability**: Manual overrides and elimination logic expose parameters rather than branching on caller-specific behaviour.

## Structure
```
property_adviser/feature/
  config.py    # Typed config schema + loader (supports multi-target batches)
  pipeline.py  # run_feature_selection orchestration + guardrails
  cli.py       # Thin CLI / batch runner
  compute.py   # Metric computation helpers
```

## Inputs
- Derived dataset: typically `data/preprocess/derived.csv`.
- Configuration: `config/features.yml`, supporting threshold and top-k modes, exclusion lists, guardrail settings, and optional RFECV elimination.
- Manual overrides can be supplied via CLI flags, GUI orchestration, or programmatic calls to `run_feature_selection`.
- Config-driven overrides: set `include_columns` / `manual_exclude` in `config/features.yml` to guarantee critical predictors (e.g., segment buckets) remain in the selected set while still computing their feature importance.

## Metrics & Selection
- Metrics: `pearson_abs`, normalised `mutual_info`, `eta`, and `bic_improvement` (computed via `compute_feature_scores`).
- `bic_improvement` = BIC(null) − BIC(model with feature), using OLS; positive means the feature improves fit. Categorical features use one-hot encoding (drop-first).
- Best score = `max(pearson_abs, mutual_info, eta, bic_improvement)` with `best_metric` storing the winner.
- Guardrails: drop ID-like columns, enforce family rules, prune highly correlated pairs, and annotate reasons in the shared scores table.
- Manual overrides (`include`, `exclude`, `use_top_k`, `top_k`) always take precedence.
- Configured `include_columns` are prepended before manual CLI overrides so critical columns survive thresholding and redundancy pruning.
- Optional RFECV elimination (configured via `elimination`): supports `max_features` (limit columns passed to RFE), `sample_rows` (row sampling for speed), and logs preprocessing + fit durations alongside summary scores.
- Multi-target runs: define `targets` in `config/features.yml` to generate per-horizon outputs (each target writes to `data/features/<target>/`).

## Outputs (`data/features/`)
- `feature_scores.parquet` (or `.csv`): full metrics, selection flags, override reasons, and elimination metadata.
- `X.<ext>` / `y.<ext>` / `training.<ext>`: feature matrix, target vector, combined dataset (format controlled by `dataset_format`).
- `selected_features.txt`: newline-separated list of selected predictors.

## CLI Usage
```bash
uv run pa-feature --config config/features.yml --scores-file feature_scores.parquet
```
- Emits structured logs (`feature_selection.start`, `.metrics`, `.complete`, `.elimination`) via `core.app_logging`.

## Programmatic Usage
```python
from property_adviser.feature import load_feature_selection_config, run_feature_selection

configs = load_feature_selection_config(Path("config/features.yml"))
for cfg in configs:
    run_feature_selection(cfg, include=[], exclude=[], write_outputs=True)
```
- Returned `FeatureSelectionResult` exposes `.scores_table`, `.selected_columns`, `.X`, and `.y` so GUIs and automation stay in sync.

## Handover to Training
- Respect manual selection flags so model training stays deterministic.
- Update `property_adviser/train/AGENTS.md` if you add new output artefacts or change dataset formats.

## Maintenance Checklist
1. Keep guardrail logic deterministic and configuration-driven; document new rules here.
2. Extend elimination support via config (new estimators, sampling, feature caps) rather than branching per caller.
3. Preserve schema compatibility when adding metrics—augment the scores table instead of renaming columns unless necessary.
