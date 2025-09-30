## Property Analyst

### Overview
End-to-end pipeline for forecasting property sale prices from suburb-level CSVs.
The workflow is modular and agent-friendly: **Macro data → Preprocessing → Feature selection → Training → Prediction**.

---

## Repository Layout
- `data/raw` – source CSVs from vendors
- `data/preprocess` – cleaned tables, segment-level features/targets, optional detail snapshots, audits, metadata
- `data/training` – feature matrices, targets, scores, selected feature lists
- `data/macro` – macroeconomic series fetched from external sources
- `models` – persisted models, score summaries
- `config` – YAMLs for every stage (`macro.yml`, `preprocessing.yml`, `pp_clean.yml`, `pp_derive.yml`, `features.yml`, `model.yml`)
- `property_adviser/` – core Python package (`core`, `macro`, `preprocess`, `feature`, `train`, `predict`) with module docs co-located as `AGENTS.md`
- `app/` – Streamlit applications (`predict_app.py`, `market_insights_app.py`) with their own `AGENTS.md`
- Module docs live alongside code as `AGENTS.md`; shared contract notes are captured in the root `AGENTS.md` playbook.

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
Produces CPI, cash-rate, and ASX index tables under `data/macro/`, plus a merged `macro_au_annual.csv` ready to join on sale year. See `property_adviser/macro/AGENTS.md` for source contracts and the `add_macro_yearly` helper.

### 1) Preprocessing
```bash
uv run pa-preprocess --config config/preprocessing.yml --verbose
```
`preprocessing.yml` orchestrates cleaning (`pp_clean.yml`) and derivation (`pp_derive.yml`). Outputs include `cleaned.parquet`, the segment dataset (`segments.parquet`) with forward targets/horizons, optional `derived_detailed.parquet`, metadata, and drop audits. Details live in `property_adviser/preprocess/AGENTS.md` (seasonality features, suburb trends, ratios, age buckets, macro joins, buckets, future targets, etc.).

### 2) Feature Selection
```bash
uv run pa-feature --config config/features.yml --verbose
```
Scores every candidate with Pearson, normalised mutual information, and correlation ratio, then applies threshold or top-k selection. Guardrails drop ID-like columns, enforce family rules, and prune highly correlated predictors while respecting manual `include` / `exclude` overrides. When multiple targets are declared, the CLI iterates through each target and writes artefacts to `data/training/<target>/`:
- `feature_scores.parquet` (full metrics + selection reasons)
- `X.parquet`, `y.parquet`, `training.parquet`
- `selected_features.txt`
See `property_adviser/feature/AGENTS.md` for configuration tips.

### 3) Model Training / Selection
```bash
uv run pa-train --config config/model.yml --verbose
```
Consumes the selected features for each configured target/horizon, applies manual overrides from the scores table, and performs month-based train/validation splits. Each enabled estimator runs through a GridSearchCV pipeline (with automatic preprocessing). Artefacts are timestamped under `models/<target>/` (`best_model_*.joblib`, `model_scores_*.csv`, `feature_metadata.json`) and a consolidated `training_report_*.json` summarises every target. Refer to `property_adviser/train/AGENTS.md` for split rules, supported models, and extension points.

### 4) Prediction
```bash
# Batch scoring handled in notebooks or orchestrations
```
Load the persisted pipeline and score new rows using helpers in `property_adviser/predict`. Refer to `property_adviser/predict/AGENTS.md` for API documentation and metadata contracts.

---

## Generated Artefacts (summary)
- **Macro**: `cpi_quarterly.csv`, `cpi_annual_*.csv`, `rba_cash_*.csv`, `asx200_yearly.csv`, `macro_au_annual.csv`
- **Preprocess**: `cleaned.parquet`, `segments.parquet`, optional `derived_detailed.parquet`, `metadata.json`, optional `dropped_rows.parquet`
- **Feature selection**: per-target outputs (`feature_scores.parquet`, `X.parquet`, `y.parquet`, `training.parquet`, `selected_features.txt`)
- **Training**: per-target artefacts (`best_model_*.joblib`, `model_scores_*.csv`, `feature_metadata.json`) plus consolidated `training_report_*.json`

---

## References
- `property_adviser/macro/AGENTS.md`
- `property_adviser/preprocess/AGENTS.md`
- `property_adviser/feature/AGENTS.md`
- `property_adviser/train/AGENTS.md`
- `property_adviser/predict/AGENTS.md`
- `property_adviser/core/AGENTS.md`
- `app/AGENTS.md`
- Root `AGENTS.md` — pipeline walkthrough plus shared conventions, schema expectations, glossary, and development standards

Optional Streamlit front-ends:
```bash
uv pip install streamlit
uv run streamlit run app/predict_app.py
uv run streamlit run app/market_insights_app.py
```
