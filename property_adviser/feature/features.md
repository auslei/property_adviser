# Feature & Target Reference

## Purpose
- Document derived fields written to `data/preprocess/segments.parquet` so downstream feature selection and modelling can reference a single contract.
- Clarify how each feature/target is calculated, including any smoothing or leakage guards.

## Feature Families

### Temporal & Seasonality
- `month_id`: Dense rank of `saleYearMonth`, offset from zero; creates a monotonic time index.
- `saleMonth_sin`, `saleMonth_cos`: Sine/cosine encodings of calendar month (`saleMonth`) to capture annual seasonality without discontinuity between December and January.

### Suburb Trend Roll-ups (lagged to prevent leakage)
- `suburb_price_median_current`: Prior-month median sale price for the suburb.
- `suburb_price_median_{3m,6m,12m}`: Rolling medians of the prior-month median over the last 3/6/12 months.
- `suburb_txn_count_current`: Prior-month transaction count for the suburb.
- `suburb_txn_count_{3m,6m,12m}`: Rolling sums of monthly transaction counts.
- `suburb_volatility_current`: Prior-month standard deviation of prices.
- `suburb_volatility_{3m,6m,12m}`: Rolling means of monthly standard deviation.
- `suburb_delta_{3m,12m}`: Percent change of the prior-month median versus 3/12 months earlier.
- Property-type scoped variants (e.g. `suburb_house_price_median_current`, `suburb_house_delta_12m`) follow the same formulas after mapping property types to tags (house/other).

### Segment-Level Snapshots
- `record_count`: Number of transactions contributing to the segment/month combination.
- `current_price_median`: Segment-level median sale price for the observation month (grouped by suburb, property type, and configured buckets).
- `current_price_median_roll_mean_12m`: 12-month rolling mean of the current segment median.
- `current_price_median_roll_std_12m`: 12-month rolling standard deviation of the current segment median.
- `current_price_median_z_12m`: Z-score of the current median vs the rolling mean/std window above.
- `current_price_median_yoy`: Year-over-year percent change (ratio vs price 12 months prior).
- `current_price_median_6m_change`: Percent change vs price 6 months prior.
- `current_price_median_rel_suburb`: Ratio of the segment median to the suburb median for the same month.

### Ratio & Density Features
- `ratio_land_per_bed`: `(landSizeM2) / (bed + 1)`; denominator offset avoids divide-by-zero when `bed` is 0.
- `ratio_floor_per_bed`: `(floorSizeM2) / (bed + 1)`.
- `ratio_car_per_bed`: `(car) / (bed + 1)`.
- `ratio_bed_bath`: `(bed) / (bath + 1)`.

### Price per Area
- `price_per_sqm_land`: `(salePrice) / landSizeM2`; rows with non-positive area are dropped.
- `price_per_sqm_floor`: `(salePrice) / floorSizeM2`; rows with non-positive area are dropped.

### Relative Pricing Signals
- `rel_price_vs_suburb_median`: Transaction price divided by the suburb median for the same month.
- `rel_price_vs_region_median`: Transaction price divided by the region-month median (when region field is present).
- `rel_street_effect`: Leave-one-out mean street price relative to the suburb mean; only populated when the street has at least 5 observations.

### Property Age & Bands
- `propertyAge`: `saleYear - yearBuilt`, clipped below `min_age` (currently 0) and optionally above `max_age` when provided.
- `propertyAgeBand`: Buckets `propertyAge` into "0–5", "6–20", and "21+" by default.

### Discrete Buckets for Segment Keys
- `bed_bucket`: Buckets bedroom counts into `<=2`, `3`, `4`, `5+`.
- `bath_bucket`: Buckets bathroom counts into `0-1`, `2`, `3`, `4+`.
- `land_bucket`: Land size buckets (`<=400`, `401-700`, `700+` square metres).
- `floor_bucket`: Floor size buckets (`<=150`, `151-220`, `220+` square metres).

### Macro Signals (annual, joined on `saleYear`)
- `cpi_index_dec`, `cpi_yoy_dec`, `cpi_index_avg`, `cpi_yoy_avg`: CPI level and growth metrics (December point-in-time and yearly averages).
- `cash_rate_avg`, `cash_rate_eoy`, `cash_rate_change_avg`, `cash_rate_change_eoy`: RBA cash rate levels and year-on-year changes.
- `asx200_close`, `asx200_yoy`: ASX 200 annual close and year-on-year growth.

## Target Family
- `price_future_6m`: Median sale price for the same segment 6 calendar months after the observation month (lagged roll with `min_periods=6`).
- `price_future_6m_diff`: `price_future_6m - current_price_median` for the observation month.
- `price_future_6m_delta`: `(price_future_6m / current_price_median) - 1`, capturing expected growth rate.
- `price_future_12m`: Median sale price 12 months ahead (lagged, `min_periods=12`).
- `price_future_12m_diff`: `price_future_12m - current_price_median`.
- `price_future_12m_delta`: `(price_future_12m / current_price_median) - 1`.
- `price_future_12m_smooth`: 12-month rolling mean of `price_future_12m`, reducing noise in sparse segments.
- `price_future_12m_smooth_diff`: `price_future_12m_smooth - current_price_median`.
- `price_future_12m_smooth_delta`: `(price_future_12m_smooth / current_price_median) - 1`; preferred for stable long-horizon models.

## Usage Notes
- All rolling features apply a one-month shift before aggregation, ensuring only historical data informs each observation (no leakage from the target month).
- When experimenting with different modelling "modes", toggle targets in `config/features.yml` / `config/model.yml` via the `enabled` flag while keeping this reference file as the single source of truth for field definitions.
