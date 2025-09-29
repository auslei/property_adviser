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
The derivation step engineers additional predictors from cleaned fields. The table below summarises the derived columns, how they are calculated, and why they exist.

| Feature(s) | Definition | Rationale |
| --- | --- | --- |
| `month_id` | Dense rank of `saleYearMonth`, optionally offset from zero. | Provides a monotonic time index for models that need an ordinal month reference. |
| `saleMonth_sin`, `saleMonth_cos` | Sine/cosine encoding of `saleMonth`. | Captures within-year seasonality without discontinuity between December → January. |
| `suburb_price_median_current` | Prior-month suburb median sale price (lagged by one month). | Anchors each record to the last known market level without leaking the current sale. |
| `suburb_price_median_3m`, `suburb_price_median_6m`, `suburb_price_median_12m` | Rolling medians of the lagged suburb median across 3/6/12 months. | Smooths medium- and long-term suburb price trends for stability. |
| `suburb_txn_count_3m`, `suburb_txn_count_6m`, `suburb_txn_count_12m` | Rolling sums of lagged suburb transaction counts over 3/6/12 months. | Proxies recent demand depth and liquidity in the suburb. |
| `suburb_volatility_3m`, `suburb_volatility_6m`, `suburb_volatility_12m` | Rolling means of lagged suburb price standard deviation across 3/6/12 months. | Measures recent market volatility to distinguish steady vs. volatile suburbs. |
| `suburb_delta_3m`, `suburb_delta_12m` | Percent change of the lagged suburb median vs. 3 and 12 months ago. | Captures short/long momentum while remaining leakage-safe. |
| `suburb_<tag>_*` (e.g. `suburb_house_price_median_current`, `suburb_other_txn_count_6m`) | Same aggregates as above, but calculated within property-type tags supplied via `type_col`. | Gives the model property-type-specific market context so houses aren’t compared to units and vice versa. |
| `land_per_bed`, `floor_per_bed`, `car_per_bed`, `bed_bath_ratio` | Ratios created via `_safe_ratio` using bed/bath counts plus one to avoid zero division. | Normalises size/amenity features by occupancy to highlight layout efficiency. |
| `price_per_sqm_land`, `price_per_sqm_floor` | Sale price divided by land/floor area with optional clipping. | Adds value-per-area signals that correlate with density and desirability. |
| `rel_price_vs_suburb_median` | `salePrice / suburb_price_median_current`. | Indicates how a sale compares to the suburb benchmark; reused at prediction for interpretability. |
| `rel_street_effect` | Leave-one-out ratio of street mean price to suburb mean (when sufficient samples). | Captures persistent street-level premiums or discounts. |
| `propertyAge` | `saleYear – yearBuilt`, constrained to non-negative values. | Reflects property vintage, a strong price driver. |
| `propertyAgeBand` | Categorical bins of `propertyAge` using configured cut-offs (default 0–5, 6–20, 21+). | Buckets age into coarse segments that play nicely with categorical models. |

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
