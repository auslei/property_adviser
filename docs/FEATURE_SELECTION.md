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
- **Derived dataset**: output of `preprocess/cli.py` (typically `data_preprocess/derived.parquet`)
- **Config** (`config/features.yml`):
  - `input_file`: path to derived dataset
  - `output_dir`: where outputs are written
  - `target`: target column name
  - `correlation_threshold`: numeric threshold (applied to best_score)
  - `exclude_columns`: columns to ignore
  - `mi_random_state`: random seed for MI estimation
  - `top_k`: (optional) if present, top-k selection mode is enabled

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

### Outputs
- **Scores + selection file** (single Parquet/CSV file, default `feature_scores.parquet`):
  - Columns: `feature, pearson_abs, mutual_info, eta, best_metric, best_score, selected, reason`
- **Datasets**:
  - `X.parquet` — selected features
  - `y.parquet` — target column
  - `training.parquet` — X + y combined (compatibility)
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
```json
{
  "event": "feature_selection.complete",
  "features": <count>,
  "output_dir": "<path>",
  "threshold": <float>,
  "use_top_k": <bool>,
  "top_k": <int or null>,
  "include": <count>,
  "exclude": <count>
}
```
