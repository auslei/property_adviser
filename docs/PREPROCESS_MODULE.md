# Preprocess Module — Documentation

This module handles the **data preparation pipeline** for property analytics.  
It is split into **cleaning** and **derivation**, each configured via YAML files.

---

## 🔧 Structure

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

---

## 🧹 Cleaning (preprocess_clean.py)

### Purpose
- Standardise messy categories (`house`, `House`, `Town House` → `House`)
- Consolidate fields into canonical forms (`agency` → `agencyBrand`)

### How it works
- One unified function `apply_category_mappings` supports:
  - **In-place mappings** (overwrite same column)
  - **New-column mappings** (map from `source` → `output`)

### Config Example (pp_clean.yml)
```yaml
category_mappings:
  propertyType:        # in-place
    House: [house, townhouse, dwelling]
    Unit:  [unit, villa, duplex]
    Apartment: [apartment, studio, flat]

  agencyBrand:         # new column
    source: agency
    default: Other
    mode: contains
    rules:
      Noel Jones:     [NOEL JONES]
      Barry Plant:    [BARRY PLANT]
      Jellis Craig:   [JELLIS CRAIG]
      Ray White:      [RAY WHITE]
      Harcourts:      [HARCOURTS]
      Woodards:       [WOODARDS]
      Biggin & Scott: [BIGGIN & SCOTT]
      Fletchers:      [FLETCHERS]
      Unknown:        [UNKNOWN]
      Other:          [OTHER]
```

---

## 🧮 Derivations (preprocess_derive.py)

### Purpose
Create **new numeric/categorical features** for modelling, based on cleaned data.

### Key functions
- **Date parts**: `saleYear`, `saleMonth`, `saleYearMonth`
- **Property category**: consolidate `propertyType` + `landUse` → `propertyCategory`
- **Ratios**: land/floor ratio, bed/bath ratio, car/bed ratio
- **Per-area prices**: price per sqm land, price per sqm floor
- **Age**: years since built
- **Market aggregates**: year-month mean, suburb-month median
- **Relative features**: ratio of salePrice to suburb-month median
- **Seasonality**: cyclical encoding of saleMonth (sin/cos)

### Config Example (pp_derive.yml)
```yaml
derivations:
  property_category:
    enabled: true
    property_type_col: propertyType
    land_use_col: landUse
    output: propertyCategory
    unknown_value: Other

  property_age:
    enabled: true
    year_built_col: yearBuilt
    sale_year_col: saleYear
    output: propertyAge
    min_age: 0

  land_floor_ratio:
    enabled: true
    numerator: landSizeM2
    denominator: floorSizeM2
    output: landFloorRatio
    clip_min: 0.01
    clip_max: 200
```

---

## 🚀 CLI (cli.py)

### Purpose
Command-line entrypoint to run cleaning + derivation in sequence.

### Usage
```bash
uv run pa-preprocess --config config/preprocessing.yml --verbose
```

### Config (preprocessing.yml)
```yaml
paths:
  cleaned_path: data_preprocess/cleaned.parquet
  derived_path: data_preprocess/derived.parquet
  metadata_path: data_preprocess/metadata.json

configs:
  clean: config/pp_clean.yml
  derive: config/pp_derive.yml

options:
  verbose: true
```

### Flow
1. Load top-level config
2. Resolve paths
3. Load sub-configs (`pp_clean.yml`, `pp_derive.yml`)
4. Run:
   - `clean_data(cfg_clean)` → save cleaned
   - `derive_features(cfg_derive)` → save derived
   - Write metadata JSON

---

## ✅ Summary
- **Cleaning** = label standardisation, mapping messy categories
- **Derivation** = computed features (ratios, aggregates, encodings)
- **CLI** orchestrates the pipeline using split configs
- **All configs YAML-driven** → flexible, reproducible, easy to extend
