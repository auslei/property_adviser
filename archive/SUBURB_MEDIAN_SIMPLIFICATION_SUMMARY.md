# Suburb Median Simplification Summary

## Overview
This document captures the implementation summary for the suburb median simplification project.

## Architectural Principles
1. **Module Responsibility**: All complex calculations should be done in corresponding modules, not in the Streamlit app. Calculations should only be performed in the Streamlit app if they are specifically for charting and simple.
2. **Data Derivation**: All data derivation and preprocessing should be done in preprocessing steps, not in other modules. This ensures derived features like medians and price factors are precomputed and available in the derived dataset.

## Changes Made

### 1. **Eliminated Complex Forecasting Model**
**Before**: Used GradientBoostingRegressor to predict medians for unseen suburb/year/month combinations
**After**: Uses only observed historical medians with global fallback for missing combinations

### 2. **Simplified Function Structure**
**Functions Removed**:
- `_train_forecaster()` - Complex ML model training
- `estimate_suburb_median()` - Forecasting-based median estimation
- `_compute_time_index()` - Time index computation for ML features
- Various helper functions for ML model management

**Functions Added/Retained**:
- `compute_baseline_medians()` - Simple historical median computation
- `prepare_suburb_median_artifacts()` - Simplified preparation with backward compatibility
- `load_baseline_median_history()` - Simplified loading of historical medians

### 3. **Reduced Dependencies**
**Before**: Required scikit-learn components (GradientBoostingRegressor, Pipeline, ColumnTransformer, OneHotEncoder)
**After**: Only requires pandas for data manipulation

### 4. **Improved Maintainability**
- Reduced code complexity from ~250 lines to ~100 lines
- Eliminated ML model serialization/deserialization
- Removed unnecessary mathematical transformations (sin/cos time encoding)
- Simplified data flow from preprocessing → median computation → feature engineering

## Technical Improvements

### 1. **More Transparent Logic**
The new approach directly computes medians from observed data:
```python
# Simple approach - no ML required
suburb_medians = df.groupby(['suburb', 'saleYear', 'saleMonth'])['salePrice'].median()
```

### 2. **Better Performance**
- Faster execution (no ML training/inference overhead)
- Lower memory footprint (no model artifacts to store/load)
- Simpler data pipeline

### 3. **Easier Debugging**
- Deterministic results based on historical data
- No black-box ML model predictions
- Clear lineage from input data to computed medians

## Backward Compatibility
The `prepare_suburb_median_artifacts()` function maintains the same interface as before to ensure existing code continues to work without modification.

## Impact on Feature Engineering
The `baselineMedian` column used in feature selection for computing `priceFactor = salePrice / baselineMedian` now uses:
1. **Observed historical medians** when available for suburb/year/month
2. **Global historical medians** as fallback when suburb-specific data is unavailable
3. **No forecasting/prediction** which was often inaccurate anyway

This approach is more reliable because it's based on actual historical data rather than ML model predictions that could be wrong.

## Files Modified
1. `src/suburb_median.py` - Completely rewritten with simplified approach
2. `src/feature_selection.py` - Updated imports and `_attach_baseline_median()` function

## Benefits
1. **Simplicity**: Much easier to understand and maintain
2. **Reliability**: Based on actual data rather than ML predictions
3. **Performance**: Faster execution without ML overhead
4. **Reduced Complexity**: Fewer dependencies and less code to maintain
5. **Alignment with Use Case**: Matches the actual requirement of using medians as denominators for price factors