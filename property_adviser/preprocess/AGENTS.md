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
  config.py            # Typed config schema + loader
  pipeline.py          # run_preprocessing orchestration + metadata builder
  cli.py               # CLI / thin wrappers around the pipeline
  preprocess_clean.py  # Cleaning stage
  preprocess_derive.py # Derivation stage
```
Supporting config lives under `config/`:
- `config/preprocessing.yml` (controller)
- `config/pp_clean.yml`
- `config/pp_derive.yml`

## Cleaning (`preprocess_clean.py`)
- Standardises categorical noise (e.g., `house`, `House`, `Town House` → `House`). Case-insensitive keyword matching supports optional defaults via `default` / `preserve_unmatched` flags so unmatched values can fall back to labels like `Unknown`.
- Renames source columns to canonical names and coerces numeric dtypes.
- Optionally records dropped rows to the configured audit location.

## Derivation (`preprocess_derive.py`)
Engineer leak-safe features using deterministic inputs:
- Seasonality encodings (`saleMonth_sin`, `saleMonth_cos`, `month_id`).
- Suburb rolling aggregates for price, volume, and volatility across 3/6/12 windows.
- Property-type scoped metrics (e.g., `suburb_house_price_median_current`).
- Ratio features (`ratio_land_per_bed`, `price_per_sqm_land`, etc.) built via `_safe_ratio`; declare them with `fn: ratio` / `fn: price_per_area` in `pp_derive.yml` and optional `denominator_offset` values to emulate the legacy “plus one” patterns safely.
- Age features (`propertyAge`, `propertyAgeBand`) with configurable buckets.
- Optional macro joins via `macro_join` config, which calls `property_adviser.macro.add_macro_yearly` to merge annual CPI/cash-rate/ASX features on `saleYear`.
- Segment builder now computes trend-aware features (`current_price_median_z_12m`, rolling means/std, YoY/6m change) and derives future diffs/deltas (e.g. `price_future_6m_delta`) so downstream models can forecast relative moves instead of raw levels.
- Future-target outputs are fully configuration-driven: each entry in `future_targets` can declare a `base_column`, a set of `derived` operations (delta/diff/ratio/copy), and optional `smooth` settings (window/min periods/derived outputs) without touching code.
- Configurable bucketing turns raw attributes into segment features (`bed_bucket`, `bath_bucket`, `land_bucket`, `floor_bucket`) using YAML-defined bins or mappings.
- Segment aggregation builds one row per observation month using the configured grouping keys (default: suburb + property type + buckets). Aggregated metrics (`current_price_median`, `transaction_count`, etc.) and lead targets (e.g., `price_future_6m`, `price_future_12m`) are computed from the raw data with look-ahead horizons. Use `aggregations.carry` to keep deterministic historical roll-ups (e.g., `suburb_price_median_6m`) without hand-writing duplicate metric entries.

## Outputs
All files are written using `property_adviser.core.io.save_parquet_or_csv`:
- `data/preprocess/cleaned.csv`
- `data/preprocess/segments.parquet` (segment-level dataset)
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
