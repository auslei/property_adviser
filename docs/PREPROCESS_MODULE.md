## Preprocess Module — Documentation

This module handles the **data preparation pipeline** and feeds the **feature selection** stage with a clean, derived dataset. It is split into **cleaning** and **derivation**, each configured via YAML files.

### Structure
```
property_adviser/
  preprocess/
    cli.py
    preprocess_clean.py
    preprocess_derive.py
  config/
    preprocessing.yml   # top-level orchestrator
    pp_clean.yml        # cleaning config
    pp_derive.yml       # derivation config
```

### Cleaning (`preprocess_clean.py`)
- Standardises categories (e.g. `house`, `House`, `Town House` → `House`).
- Consolidates fields into canonical forms (e.g. `agency` → `agencyBrand`).

### Derivations (`preprocess_derive.py`)
- Numeric/categorical features:
  - Date parts (`saleYear`, `saleMonth`, `saleYearMonth`)
  - Property category (consolidates `propertyType` + `landUse`)
  - Ratios (land/floor, bed/bath, car/bed)
  - Per-area prices (land/floor)
  - Age (years since built)
  - Market aggregates (year-month mean, suburb-month median)
  - Relative features (price vs suburb-month median)
  - Seasonality encodings (sin/cos of month)

### CLI (`preprocess/cli.py`)
- Orchestrates cleaning + derivation from `config/preprocessing.yml`.
- Writes:
  - `data_preprocess/cleaned.parquet`
  - `data_preprocess/derived.parquet`
  - `data_preprocess/metadata.json`
- **Downstream contract**:
  - Feature selection reads **derived.parquet** and the target listed in `config/features.yml`.

### Notes for Feature Selection (handover)
- Derived dataset should contain:
  - Target column (as specified in `features.yml`)
  - Predictors with proper dtypes (numeric/categorical).
- Avoid leakage: only derived features built from training-time information.
