## Feature Selection Module — Documentation

This module performs **feature scoring and selection** on the derived dataset, producing
X and y matrices for training as well as a comprehensive scores table for inspection.

### Structure
```
property_adviser/
  feature/
    compute.py       # compute per-feature metrics
    selector.py      # threshold/top-k selection logic
    cli.py           # CLI and reusable entry point
```

### Inputs
- **Derived dataset**: output of `preprocess/cli.py` (typically `data/preprocess/derived.csv`)
- **Config** (`config/features.yml`):
  - `input_file`: path to derived dataset
  - `output_dir`: where outputs are written
  - `target`: target column name
  - `correlation_threshold`: numeric threshold (applied to best_score)
  - `exclude_columns`: columns to always ignore (e.g. IDs such as `parcelDetail`, addresses)
  - `mi_random_state`: random seed for MI estimation
  - `top_k`: (optional) if present, top-k selection mode is enabled
  - `elimination`: optional block to enable recursive feature elimination (see below)

> **Tip:** Use `exclude_columns` for dropping ID-like fields (unique identifiers such as `parcelDetail`, `streetAddress`, etc.).
> These columns won’t be scored or considered for selection, and their exclusion reason will be recorded as
> `manual exclude (not selected)` in the scores file.

### Methods
- **Metrics per feature**:
  - `pearson_abs` — absolute Pearson correlation with target (numeric)
  - `mutual_info` — mutual information with target (numeric, min–max **normalised** per run)
  - `eta` — correlation ratio for categorical → numeric association
- **Best score logic**:
  - `best_score = max(pearson_abs, mutual_info, eta)`
  - `best_metric = argmax(...)`

### Selection Logic
1. **Threshold mode** (default): select if `best_score >= correlation_threshold`.
2. **Top-k mode** (if `top_k` present or GUI forces it): rank by `best_score` and keep top-k features.
3. **Manual overrides** (GUI-friendly):
   - `include`: always select, reason = "manual include"
   - `exclude`: never select, reason = "manual exclude (not selected)"
   - `use_top_k`: None = follow config; True/False = force
   - `top_k`: override value
4. **Guardrails**: drop ID-like columns, apply family keep rules, prune highly correlated pairs.
5. **Recursive Feature Elimination (optional)**: if `elimination.enable` is true, run RFECV with the configured estimator on the surviving columns to produce a model-aware subset. Features that survive elimination are marked with `elimination_selected = True` in the scores table; the minimum RFE rank is stored in `elimination_rank`.

### Recursive Feature Elimination configuration
The `elimination` block accepts:

| Key | Description | Default |
| --- | --- | --- |
| `enable` | Turn the elimination step on/off. | `false` |
| `estimator` | Estimator name (`RandomForestRegressor`, `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`). | `RandomForestRegressor` |
| `estimator_params` | Dict of keyword arguments passed to the estimator constructor. | `{}` |
| `step` | Number of features to drop per RFE iteration. | `1` |
| `min_features` | Minimum original features to keep (before one-hot expansion). | `5` |
| `scoring` | Sklearn scoring string used by RFECV. | `r2` |
| `cv` | Cross-validation folds for RFECV. | `3` |
| `n_jobs` | Parallelism for RFECV (pass `-1` for all cores). | `None` |

The step uses a preprocessing pipeline (`SimpleImputer` + `StandardScaler` + `OneHotEncoder`) so tree and linear models can work with mixed datatypes. Manual include lists still win—anything in `include` is re-added after RFE even if the estimator would drop it.

### Outputs
- **Scores + selection file** (single Parquet/CSV file, default `feature_scores.parquet`):
  - Columns: `feature, pearson_abs, mutual_info, eta, best_metric, best_score, selected, reason, elimination_rank, elimination_selected`
- **Datasets**:
  - `X.csv` — selected features (Parquet if you change the extension)
  - `y.csv` — target column
  - `training.csv` — X + y combined (compatibility)
- **Other files**:
  - `selected_features.txt` — plain list of selected column names

### CLI
```bash
uv run pa-feature --config config/features.yml --scores-file feature_scores.parquet
```

### Programmatic Usage
```python
from property_adviser.feature.cli import run_feature_selection
from property_adviser.core.config import load_config

cfg = load_config("config/features.yml")
result = run_feature_selection(
    cfg,
    include=[],            # manual include
    exclude=[],            # manual exclude
    use_top_k=None,        # None: follow config; True/False to force
    top_k=None,            # optional override if top-k is used
    write_outputs=False
)

# Access results
scores = result.scores_table   # DataFrame with all metrics + selection info
X, y = result.X, result.y
selected = result.selected_columns
```

### Logging
On completion, an event is logged:
`feature_selection.complete` still fires with the final feature count. When elimination runs successfully an additional `feature_selection.elimination` event is emitted (`estimator`, `selected`, `total`, `best_score`, `step`, `cv`).
