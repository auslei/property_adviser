## Property Analyst

### Overview
End-to-end pipeline for forecasting property sale prices from suburb-level CSVs.  
The pipeline is modular: **Preprocessing → Feature Selection → Training → Prediction**. The Streamlit app provides an interactive UI.

---

## Repository Layout
- `data/` – raw inputs
- `data_preprocess/` – cleaned + derived datasets, metadata
- `data_training/` – training artefacts (X, y, splits), feature metadata
- `models/` – fitted models + metrics
- `config/` – YAML configurations (`preprocessing.yml`, `pp_clean.yml`, `pp_derive.yml`, `features.yml`, `model.yml`)
- `property_adviser/`:
  - `core/` (logging, IO, utils)
  - `preprocess/` (clean, derive, CLI)
  - `feature/` (feature selection)
  - `train/` (model training)
  - `predict/` (prediction)
- `app/` — Streamlit UI pages

---

## Environment
```bash
uv venv && source .venv/bin/activate
uv sync
# or pip install -r requirements.txt
```

---

## Running the Pipeline

### 1) Preprocessing
```bash
uv run pa-preprocess --config config/preprocessing.yml --verbose
```
Writes:
- `data_preprocess/cleaned.parquet`
- `data_preprocess/derived.parquet`
- `data_preprocess/metadata.json`  
See PREPROCESS_MODULE.md for details.

### 2) Feature Selection (now GUI-friendly)
Full module docs: FEATURE_SELECTION.md

**CLI:**
```bash
uv run pa-feature --config config/features.yml --scores-file feature_scores.parquet
```
**What it does:**
- Computes per-feature metrics: `pearson_abs`, **normalised** `mutual_info` [0–1], `eta`.
- Builds `best_score = max(pearson_abs, mutual_info, eta)` and `best_metric`.
- Selection modes:
  - **Threshold**: `best_score >= correlation_threshold`
  - **Top-k**: rank by `best_score` and keep `k` (enabled if `top_k` present or forced)
- **Manual overrides**: `include` (always select), `exclude` (never select)
- **Outputs**:
  - `data_training/X.parquet`, `data_training/y.parquet`
  - `data_training/training.parquet` (X + y combined)
  - **Single scores+selection file** (by default `data_training/feature_scores.parquet`) with columns:  
    `feature, pearson_abs, mutual_info, eta, best_metric, best_score, selected, reason`

**Programmatic (for GUI):**
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

scores = result.scores_table   # DataFrame with metrics, selected, reason
X, y = result.X, result.y
selected = result.selected_columns
```

### 3) Model Training/Selection
```bash
uv run pa-train --config config/model.yml
```
See TRAINING.md for configuration options, validation strategy, and artefact outputs.


### 4) Prediction
```bash
uv run pa-predict --input new_data.csv --model models/best_model.pkl
```

---

## Streamlit
```bash
streamlit run app/Overview.py
```
- Preprocess data, inspect features
- Run feature selection with overrides (`include/exclude`, top-k)
- Train/evaluate models
- Predict on new data

---

## Generated Artefacts (summary)
- **Preprocess**: `cleaned.parquet`, `derived.parquet`, `metadata.json`
- **Features**: `X.parquet`, `y.parquet`, `training.parquet`, `feature_scores.parquet`
- **Models**: `best_model.pkl`, `model_metrics.json`

---

## References
- PREPROCESS_MODULE.md — preprocess pipeline details
- FEATURE_SELECTION.md — feature scoring and selection pipeline
- TRAINING.md — model training orchestration and artefacts
- COMMON.md — shared conventions, schema expectations, glossary
- DEV_GUIDELINES.md — coding/agentic standards
- AGENTS.md — pipeline overview and Streamlit pages
