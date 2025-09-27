## Property Analyst

### Overview
End-to-end pipeline for forecasting property sale prices from suburb-level CSVs.
The workflow is modular and agent-friendly: **Macro data → Preprocessing → Feature selection → Training → Prediction**.

---

## Repository Layout
- `data/raw` – source CSVs from vendors
- `data/preprocess` – cleaned tables, derived features, drop audits, metadata
- `data/training` – feature matrices, targets, scores, selected feature lists
- `data/macro` – macroeconomic series fetched from external sources
- `models` – persisted models, score summaries
- `config` – YAMLs for every stage (`macro.yml`, `preprocessing.yml`, `pp_clean.yml`, `pp_derive.yml`, `features.yml`, `model.yml`)
- `property_adviser/` – core Python package (`core`, `macro`, `preprocess`, `feature`, `train`, `predict`)
- `docs/` – deep-dive module notes (Macro, Preprocess, Feature selection, Training, etc.)

> Note: the legacy Streamlit UI has been retired; use the CLI entry points or build your own presentation layer on top of the core package.

---

## Environment
```bash
uv venv && source .venv/bin/activate
uv sync
# or pip install -r requirements.txt
```

---

## Running the Pipeline

### 0) Macro Data (optional but recommended)
```bash
uv run pa-macro --config config/macro.yml --verbose
```
Produces CPI, cash-rate, and ASX index tables under `data/macro/`, plus a merged `macro_au_annual.csv` ready to join on sale year. See `docs/MACRO.md` for column definitions and the `add_macro_yearly` helper.

### 1) Preprocessing
```bash
uv run pa-preprocess --config config/preprocessing.yml --verbose
```
`preprocessing.yml` orchestrates cleaning (`pp_clean.yml`) and derivation (`pp_derive.yml`). Outputs:
- `data/preprocess/cleaned.csv`
- `data/preprocess/derived.csv`
- `data/preprocess/metadata.json`
- Optional drop audit if `dropped_rows_path` is configured
Details live in `docs/PREPROCESS_MODULE.md` (seasonality features, suburb trends, ratios, age buckets, macro joins, etc.).

### 2) Feature Selection
```bash
uv run pa-feature --config config/features.yml --scores-file feature_scores.parquet
```
Scores every candidate with Pearson, normalised mutual information, and correlation ratio, then applies threshold or top-k selection. Guardrails drop ID-like columns, enforce family rules, and prune highly correlated predictors while respecting manual `include` / `exclude` overrides. CLI and programmatic entrypoints both return `X`, `y`, selected columns, and a rich `reason` log. Outputs land in `data/training/`:
- `feature_scores.parquet` (full metrics + selection reasons)
- `X.csv`, `y.csv`, `training.csv`
- `selected_features.txt`
See `docs/FEATURE_SELECTION.md` for configuration tips.

### 3) Model Training / Selection
```bash
uv run pa-train --config config/model.yml --verbose
```
Consumes the selected features, honours manual overrides from the scores table, and performs a month-based train/validation split. Each enabled estimator runs through a GridSearchCV pipeline (with automatic preprocessing). Optional `log_target: true` trains on log(price) and back-transforms predictions. Artefacts are timestamped under `models/`:
- `best_model_<model>_<timestamp>.joblib`
- `model_scores_<timestamp>.csv`
- `feature_metadata.json`
Refer to `docs/TRAINING.md` for split rules, supported models, and extension points.

### 4) Prediction
```bash
uv run pa-predict --input new_data.csv --model models/best_model_<...>.joblib
```
Loads the persisted pipeline and scores new rows (outside this README’s scope).

---

## Generated Artefacts (summary)
- **Macro**: `cpi_quarterly.csv`, `cpi_annual_*.csv`, `rba_cash_*.csv`, `asx200_yearly.csv`, `macro_au_annual.csv`
- **Preprocess**: `cleaned.csv`, `derived.csv`, `metadata.json`, optional `dropped_rows` parquet
- **Feature selection**: `feature_scores.parquet`, `X.csv`, `y.csv`, `training.csv`, `selected_features.txt`
- **Training**: timestamped `best_model_*.joblib`, `model_scores_*.csv`, `feature_metadata.json`

---

## References
- `docs/MACRO.md` — macro data fetcher and integration helper
- `docs/PREPROCESS_MODULE.md` — cleaning + derivation details
- `docs/FEATURE_SELECTION.md` — scoring, guardrails, and overrides
- `docs/TRAINING.md` — model orchestration and artefact schema
- `docs/PREDICT.md` — runtime prediction helpers and metadata contract

Optional Streamlit front-end:
```bash
uv pip install streamlit
streamlit run app/predict_app.py
```
- `docs/COMMON.md` — shared conventions, schema expectations, glossary
- `docs/DEV_GUIDELINES.md` — coding standards and agent workflows
- `docs/AGENTS.md` — agent-facing walkthrough of the pipeline and UI
