# Feature Selection Module Agent Guide

## Purpose & Scope
- Score and curate predictors from the derived dataset, producing training-ready feature matrices and transparent audit artefacts.
- Provide a stable CLI (`pa-feature`) and a reusable `run_feature_selection` entry point for automation.
- Reuse shared utilities from `property_adviser.core` (config loading, IO, logging) to maintain consistency across modules.

## Design Commitments
- **Clear interface**: Inputs consist of the derived dataset and `config/features.yml`; outputs are written to `data/training/` using predictable filenames.
- **High cohesion**: Scoring, guardrails, and selection logic live inside this package. When functionality becomes broadly useful (e.g., correlation helpers), promote it to `core`.
- **Low coupling**: Training and apps consume the exported files or returned dataclasses without depending on implementation details.
- **Reusability**: Manual overrides and elimination logic expose parameters rather than branching on caller-specific behaviour.

## Structure
```
property_adviser/feature/
  cli.py        # CLI / batch entry point
  compute.py    # Metric computation
  selector.py   # Guardrails, overrides, elimination
```

## Inputs
- Derived dataset: typically `data/preprocess/derived.csv`.
- Configuration: `config/features.yml`, supporting threshold and top-k modes, exclusion lists, guardrail settings, and optional RFECV elimination.
- Manual overrides can be supplied via CLI flags, GUI orchestration, or programmatic calls to `run_feature_selection`.

## Metrics & Selection
- Metrics: `pearson_abs`, normalised `mutual_info`, and `eta`.
- Best score = `max(pearson_abs, mutual_info, eta)`; `best_metric` tracks the governing metric.
- Guardrails: drop ID-like columns, enforce family keep rules, prune highly correlated pairs, and annotate reasons.
- Manual overrides: `include`, `exclude`, `use_top_k`, `top_k` always win over automatic decisions.
- Optional RFECV elimination: configured under the `elimination` block; surviving features are marked via `elimination_rank` and `elimination_selected`.

## Outputs (`data/training/`)
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
from property_adviser.feature.cli import run_feature_selection
from property_adviser.core.config import load_config

cfg = load_config("config/features.yml")
result = run_feature_selection(cfg, include=[], exclude=[], use_top_k=None, top_k=None, write_outputs=False)
```
- Returned object exposes `.scores_table`, `.X`, `.y`, and `.selected_columns` to keep GUIs and automation in sync.

## Handover to Training
- Respect manual selection flags so model training stays deterministic.
- Update `property_adviser/train/AGENTS.md` if you add new output artefacts or change dataset formats.

## Maintenance Checklist
1. Keep guardrail logic deterministic and configuration-driven; document new rules here.
2. Extend elimination support via config (e.g., new estimators) rather than branching per caller.
3. Preserve schema compatibility when adding metrics—augment the scores table instead of renaming columns unless necessary.
