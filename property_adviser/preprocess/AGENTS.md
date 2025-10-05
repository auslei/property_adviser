# Preprocess Module Agent Guide

## Purpose & Scope
- Transform raw property transactions into a feature-ready dataset by running deterministic cleaning and derivation steps.
- Offer a unified interface through typed configs (`PreprocessConfig`) and `run_preprocessing`, keeping CLIs and automation aligned.
- Lean on `property_adviser.core` for configuration, logging, IO, and step orchestration.

## Design Commitments
- **Clear interface**: The CLI accepts a single YAML (`config/preprocessing.yml`) and writes all artefacts to `data/preprocess/`. Programmatic entry points accept DataFrames/paths rather than implicit globals.
- **High cohesion**: Cleaning and derivation responsibilities live inside this package; new feature engineering should be added here unless it is reusable across modules (then promote it to `core`).
- **Low coupling**: Downstream stages consume the written artefacts; they do not call into private helpers. Upstream data sources interact only via the documented input schema.
- **Reusability**: Shared transformations (e.g., safe ratios, seasonality utilities) should be implemented once and imported across cleaning/derivation steps.

## Structure
```
property_adviser/preprocess/
  clean/
    __init__.py
    engine.py          # cleaning implementation (formerly preprocess_clean)
  derive/
    __init__.py
    config.py          # new derive spec parsing (spec_version 1)
    engine.py          # declarative driver / registry
    legacy.py          # legacy imperative implementation (spec_version 0)
    steps/
      base.py          # context + base classes
      simple.py        # expressions, mappings, street parsing, etc.
      aggregate.py     # group-level aggregations
      time_aggregate.py# windowed past/future metrics
      join.py          # dataset joins via shared IO
      binning.py       # fixed/mapping buckets
      rolling.py       # grouped rolling windows
  config.py            # typed PreprocessConfig loader
  pipeline.py          # run_preprocessing orchestration + metadata builder
  cli.py               # CLI / thin wrappers around the pipeline
  preprocess_clean.py  # compatibility re-export
  preprocess_derive.py # compatibility wrappers around engine/legacy
```
Supporting config lives under `config/`:
- `config/preprocessing.yml` (controller)
- `config/pp_clean.yml`
- `config/pp_derive.yml` (spec_version 1 declarative steps)
- `config/pp_derive_legacy.yml` (legacy derive DSL)

## Cleaning (`clean/engine.py`)
- Standardises categorical noise (e.g., `house`, `House`, `Town House` → `House`). Case-insensitive keyword matching supports optional defaults via `default` / `preserve_unmatched` flags so unmatched values can fall back to labels like `Unknown`.
- Renames source columns to canonical names and coerces numeric dtypes.
- Optionally records dropped rows to the configured audit location.
- Programmatic entry points: `clean.clean_data(...)` for inline pipelines, `clean.run_cleaning_stage(cfg)` for config-driven execution.

## Derivation (`derive/engine.py`)
Engineer leak-safe features using deterministic inputs through declarative steps:
- **Simple**: expressions (`method: expr`), categorical mappings, month index / cyclical encodings, and property-age helpers reuse the legacy implementations while exposing configuration-first APIs.
- **Aggregate**: group-level aggregations with optional `min_count` guards populate metrics such as suburb medians/counts without imperative loops.
- **Time Aggregate**: past/future windows expressed via `{past, future, unit}` produce rolling medians, momentum deltas, and future targets (`price_future_6m`, etc.) while respecting leakage controls (`include_current: false` by default).
- **Join**: standardized dataset joins resolve aliases declared in `settings.datasets` (e.g., macro CSVs) and keep left-hand columns stable.
- **Bin**: fixed or mapping-based buckets replace bespoke bucketing helpers; supply edges/labels via YAML.
- **Rolling**: grouped rolling windows (mean/median/std/etc.) deliver smoothed targets and volatility measures for downstream modelling.

### Spec versions
- **spec_version 1** (`config/pp_derive.yml`): preferred declarative format. Steps execute in-order, enabling fine-grained control over dependencies (e.g., aggregate → expression → rolling). The engine collects artefacts and applies optional `settings.default_fillna` at the end.
- **Legacy** (`config/pp_derive_legacy.yml`): still supported through `derive/legacy.py` to keep existing pipelines and tests stable while teams migrate step-by-step. Run configs without `spec_version` continue to pass through the legacy code path, including segment generation via `build_segments`.

### Adding a new step
1. Extend `derive/steps/` with a focussed implementation (subclass `DeriveStep`).
2. Register it in `derive/steps/__init__.py` so the engine can instantiate it.
3. Document the new `type` (and required fields) in `AGENTS.md` along with sample YAML.
4. Add targeted unit coverage (see `tests/test_preprocess_derive.py::test_new_spec_expression_and_aggregate`).

## Outputs
All files are written using `property_adviser.core.io.save_parquet_or_csv`:
- `data/preprocess/cleaned.csv`
- `data/preprocess/segments.parquet` (segment-level dataset; optional for spec v1 unless a dedicated segment step runs)
- `data/preprocess/derived_detailed.parquet` (optional property-level snapshot)
- `data/preprocess/metadata.json`
- Optional `data/preprocess/dropped_rows.parquet`

Schemas and dtype expectations are defined in `docs/COMMON.md`; update that contract if you alter outputs.

## Interfaces
- `load_preprocess_config(path)` → `PreprocessConfig`
- `run_preprocessing(config, write_outputs=True)` → `PreprocessResult`
- Back-compat helper `preprocess(mapping)` ingests an already-loaded YAML mapping and returns the derived dataset path.

## CLI
```bash
uv run python -m property_adviser.preprocess.cli --config config/preprocessing.yml --verbose
# or
uv run pa-preprocess --config config/preprocessing.yml --verbose
```
- The CLI wires logging via `core.app_logging`, loads typed config, and persists artefacts using the shared pipeline implementation.

## Handover to Feature Selection
- Derived dataset must include the target listed in `config/features.yml` and all engineered predictors with consistent dtypes.
- Avoid leakage by sourcing only historic aggregates; ensure new features respect the month split used during training.
- Update `property_adviser/feature/AGENTS.md` if new predictors introduce dependencies or contract changes.

## Maintenance Checklist
1. Keep configuration-driven behaviour; new toggles belong in YAML with documented defaults.
2. Share reusable helpers through `property_adviser.core` to avoid duplication across modules.
3. Add lightweight unit coverage for complex derivations and document any breaking schema changes here.

### TODO
- Re-evaluate enabling the `region_price_current` / `rel_price_region` steps once a reliable `region` column is available upstream.
- Consider wrapping the legacy relative-pricing helper as a first-class derive step to simplify the YAML expressions.
