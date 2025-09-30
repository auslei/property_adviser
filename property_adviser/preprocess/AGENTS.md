# Preprocess Module Agent Guide

## Purpose & Scope
- Transform raw property transactions into a feature-ready dataset by running deterministic cleaning and derivation steps.
- Offer a unified interface through `preprocess/cli.py` and a small set of helper functions consumed by notebooks or automation.
- Lean on `property_adviser.core` for configuration, logging, IO, and step orchestration.

## Design Commitments
- **Clear interface**: The CLI accepts a single YAML (`config/preprocessing.yml`) and writes all artefacts to `data/preprocess/`. Programmatic entry points accept DataFrames/paths rather than implicit globals.
- **High cohesion**: Cleaning and derivation responsibilities live inside this package; new feature engineering should be added here unless it is reusable across modules (then promote it to `core`).
- **Low coupling**: Downstream stages consume the written artefacts; they do not call into private helpers. Upstream data sources interact only via the documented input schema.
- **Reusability**: Shared transformations (e.g., safe ratios, seasonality utilities) should be implemented once and imported across cleaning/derivation steps.

## Structure
```
property_adviser/preprocess/
  cli.py               # Orchestrates the pipeline, handles config + logging
  preprocess_clean.py  # Cleaning stage
  preprocess_derive.py # Derivation stage
```
Supporting config lives under `config/`:
- `config/preprocessing.yml` (controller)
- `config/pp_clean.yml`
- `config/pp_derive.yml`

## Cleaning (`preprocess_clean.py`)
- Standardises categorical noise (e.g., `house`, `House`, `Town House` → `House`).
- Renames source columns to canonical names and coerces numeric dtypes.
- Optionally records dropped rows to the configured audit location.

## Derivation (`preprocess_derive.py`)
Engineer leak-safe features using deterministic inputs:
- Seasonality encodings (`saleMonth_sin`, `saleMonth_cos`, `month_id`).
- Suburb rolling aggregates for price, volume, and volatility across 3/6/12 windows.
- Property-type scoped metrics (e.g., `suburb_house_price_median_current`).
- Ratio features (`land_per_bed`, `price_per_sqm_land`, etc.) built via `_safe_ratio`.
- Age features (`propertyAge`, `propertyAgeBand`) with configurable buckets.
- Optional macro joins via `property_adviser.macro.add_macro_yearly`.

## Outputs
All files are written using `property_adviser.core.io.save_parquet_or_csv`:
- `data/preprocess/cleaned.csv`
- `data/preprocess/derived.csv`
- `data/preprocess/metadata.json`
- Optional `data/preprocess/dropped_rows.parquet`

Schemas and dtype expectations are defined in `docs/COMMON.md`; update that contract if you alter outputs.

## CLI
```bash
uv run python -m property_adviser.preprocess.cli --config config/preprocessing.yml --verbose
# or
uv run pa-preprocess --config config/preprocessing.yml --verbose
```
- The CLI wires logging via `core.app_logging` and validates artefact paths before execution.

## Handover to Feature Selection
- Derived dataset must include the target listed in `config/features.yml` and all engineered predictors with consistent dtypes.
- Avoid leakage by sourcing only historic aggregates; ensure new features respect the month split used during training.
- Update `property_adviser/feature/AGENTS.md` if new predictors introduce dependencies or contract changes.

## Maintenance Checklist
1. Keep configuration-driven behaviour; new toggles belong in YAML with documented defaults.
2. Share reusable helpers through `property_adviser.core` to avoid duplication across modules.
3. Add lightweight unit coverage for complex derivations and document any breaking schema changes here.
