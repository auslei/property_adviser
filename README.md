# Property Analyst

## Overview
End-to-end pipeline for forecasting next-year property sale prices from suburb-level CSVs.  
The pipeline is fully modular:

1. **Preprocessing** — clean messy categories and derive new features
2. **Feature Selection** — select useful predictors
3. **Model Training/Selection** — fit multiple regressors and pick the best
4. **Prediction** — score new property data using the persisted model

All artefacts (cleaned/derived data, training matrices, models, metrics) are written to disk, and a **Streamlit app** provides an interactive interface.

---

## Repository Layout
- `data/` – raw suburb-level CSV exports (inputs).
- `data_preprocess/` – cleaned + derived datasets, plus metadata.
- `data_training/` – training matrices (`X`, `y`), splits, feature metadata/importances.
- `models/` – fitted model artefacts and metrics.
- `config/` – YAML configurations:
  - `preprocessing.yml` (orchestrator; points to `pp_clean.yml`, `pp_derive.yml`)
  - `pp_clean.yml` (cleaning rules)
  - `pp_derive.yml` (derivation rules)
  - `features.yml` (feature selection)
  - `model.yml` (model training)
- `property_adviser/` – Python package with modules:
  - `core/` (logging, IO, utils)
  - `preprocess/` (clean, derive, CLI)
  - `feature/` (feature selection)
  - `train/` (model training)
  - `predict/` (prediction)
- `app/` – Streamlit UI with overview and workflows.

---

## Environment Setup
- Python 3.10+  
- Recommended: use [uv](https://github.com/astral-sh/uv) for environment management.

```bash
uv venv
source .venv/bin/activate
uv sync
```

If using `pip` directly:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Preprocessing
Run cleaning + derivations:
```bash
uv run pa-preprocess --config config/preprocessing.yml --verbose
```

This reads `pp_clean.yml` and `pp_derive.yml`, applies cleaning and derivation steps, and writes:
- `data_preprocess/cleaned.parquet`
- `data_preprocess/derived.parquet`
- `data_preprocess/metadata.json`

👉 For details of the preprocess module, see [PREPROCESS_MODULE.md](PREPROCESS_MODULE.md).

### 2. Feature Selection
```bash
uv run pa-feature --config config/features.yml
```
Persists `X.parquet`, `y.parquet`, splits, and feature metadata.

### 3. Model Training/Selection
```bash
uv run pa-train --config config/model.yml
```
Trains multiple regressors, evaluates metrics, and saves the best model under `models/`.

### 4. Prediction
```bash
uv run pa-predict --input new_data.csv --model models/best_model.pkl
```
Generates predictions using the trained model.

---

## Streamlit Application
Launch the interactive app:
```bash
streamlit run app/Overview.py
```

### Pages
- **Overview** — filters, summaries, maps
- **Data Preprocessing** — run pipeline, inspect cleaned/derived data
- **Feature Engineering** — explore correlations, feature importances, manage selections
- **Model Selection** — run training, inspect metrics
- **Prediction** — (to be added) interactive scoring

---

## Configuration
- Preprocessing:
  - `config/preprocessing.yml` → points to `pp_clean.yml`, `pp_derive.yml`
- Features: `config/features.yml`
- Models: `config/model.yml`
- Optional: `config/street_coordinates.csv` for map plots

---

## Generated Artefacts
- **Preprocess**: `cleaned.parquet`, `derived.parquet`, `metadata.json`
- **Training**: `X.parquet`, `y.parquet`, splits, feature metadata
- **Models**: `best_model.pkl`, `best_model.json`, `model_metrics.json`

---

## Maintenance
- Add/replace raw CSVs in `data/`
- Rerun preprocessing + training
- Ensure `models/best_model.pkl` matches latest metrics before using Streamlit

---

## References
- [PREPROCESS_MODULE.md](PREPROCESS_MODULE.md) — preprocess module details
- [DEV_GUIDELINES.md](DEV_GUIDELINES.md) — coding and development standards
- [AGENTS.md](AGENTS.md) — overview of agentic components
