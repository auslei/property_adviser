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
The derivation step engineers additional predictors from cleaned fields. Key groups:

#### Temporal / Market Trend Features
- `month_id`: monotonic month index (dense rank of `saleYearMonth`).
- Seasonality: cyclic sine/cosine encodings of `saleMonth`.
- Suburb-level rolling aggregates (all shifted to avoid leakage):
  - `suburb_price_median_3m`, `6m`, `12m`
  - `suburb_txn_count_3m`, `6m`, `12m`
  - `suburb_volatility_3m`, `6m`, `12m`
- Momentum: % change in suburb medians (`suburb_delta_3m`, `suburb_delta_12m`).

#### Property Size & Density Ratios
- `land_per_bed = landSize / (bed+1)`
- `floor_per_bed = propertySize / (bed+1)`
- `car_per_bed = car / (bed+1)`
- `bed_bath_ratio = bed / (bath+1)`
- `price_per_sqm_land = salePrice / landSize`
- `price_per_sqm_floor = salePrice / propertySize`
  - Division-by-zero and extreme skew handled in cleaning.

#### Relative Pricing Features
- `rel_price_vs_suburb_median = salePrice / suburb_price_median_current`
- Optional: `rel_price_vs_region_median` if a region index is available.
- `rel_street_effect`: mean street price ÷ suburb mean (leave-one-out, only if street sample size above threshold).

#### Age & Vintage
- `propertyAge = saleYear – buildYear`
- Banded into categories: `0–5`, `6–20`, `21+`.

### CLI (`preprocess/cli.py`)
- Orchestrates cleaning + derivation from `config/preprocessing.yml`.
- Writes:
  - `data/preprocess/cleaned.csv` (extension follows config)
  - `data/preprocess/derived.csv`
  - `data/preprocess/metadata.json`
- **Downstream contract**:
  - Feature selection reads the derived dataset (CSV or Parquet) and the target listed in `config/features.yml`.

### Notes for Feature Selection (handover)
- Derived dataset should contain:
  - Target column (as specified in `features.yml`)
  - Predictors with proper dtypes (numeric/categorical).
- Avoid leakage: only derived features built from training-time information.
