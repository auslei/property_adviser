# Macro Module Agent Guide

## Purpose & Scope
- Provide a cohesive ingestion layer for Australian macroeconomic indicators (CPI, cash rate, ASX index) required by downstream modelling stages.
- Expose a narrow public surface: the CLI (`pa-macro`) for batch refreshes and `add_macro_yearly` for feature joins.
- Depend on `property_adviser.core` for logging, configuration loading, and IO, keeping the module thin and reusable.

## Design Commitments
- **Clear interface**: CLI accepts a single YAML config; Python helpers require explicit paths and return pandas DataFrames with documented schemas.
- **High cohesion**: data acquisition, parsing, transformations, and file persistence are contained within `property_adviser.macro`.
- **Low coupling**: downstream modules consume outputs via files or the documented helper; no module reaches inside macro internals.
- **Reusability**: common utilities live in `property_adviser.core` (logging, config, IO) and are imported rather than reimplemented.

## Key Entry Points
- CLI: `uv run pa-macro --config config/macro.yml --verbose`
- Programmatic join: `property_adviser.macro.macro_data.add_macro_yearly(df, macro_path="data/macro/macro_au_annual.csv")`

## Configuration (`config/macro.yml`)
```yaml
start_year: 1990
outdir: data/macro
sources:
  rba_cpi_csv_url: "https://www.rba.gov.au/statistics/tables/csv/g1-data.csv"
  rba_cash_csv_url: "https://www.rba.gov.au/statistics/tables/csv/f1.1-data.csv"
  asx200_ticker: "^AXJO"    # Override to "^AORD" for longer history
```
- Keep configuration minimal and declarative so automation and analysts run the same pipeline.

## Outputs
All artefacts are written to `data/macro/` using helpers from `property_adviser.core.io`:
- `cpi_quarterly.csv`
- `cpi_annual_avg.csv`
- `cpi_annual_december.csv`
- `asx200_yearly.csv`
- `rba_cash_daily.csv`
- `rba_cash_annual.csv`
- `macro_au_annual.csv` (merged dataset used by preprocessing)

Each file ships with a stable column schema; update `COMMON.md` if you add or rename fields.

## Integration Notes
- Preprocessing joins macro features via `saleYear` using `add_macro_yearly`.
- When expanding macro coverage (e.g., new indices), keep derivations in this module and surface final metrics through the merged annual dataset to avoid leaking implementation details.

## Validation & Logging
- Logging uses `property_adviser.core.app_logging` to emit `macro.fetch`, `macro.transform`, and `macro.save` events for traceability.
- Unit tests should patch network calls and assert that outputs honour the documented schemas.

## Maintenance Checklist
1. Extend configuration defaults instead of hardcoding values in code.
2. Keep CLI arguments aligned with config keys so downstream automation remains stable.
3. Document any new outputs or breaking changes in this file and in `AGENTS.md` at the repository root.
