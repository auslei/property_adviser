## Streamlit Apps Overview

This document lists the Streamlit experiences that ship with the repository, how they fit into the
pipeline, and the artefacts they depend on.

### Property Price Predictor (`app/predict_app.py`)

| Topic | Details |
| --- | --- |
| **Purpose** | Interactive UI that lets analysts adjust property attributes and retrieve the latest ML estimate (including an optional confidence interval). |
| **Entry Point** | `uv run streamlit run app/predict_app.py` |
| **Upstream Artefacts** | - Derived dataset (streets/suburbs): resolved via `property_adviser.predict.feature_store.feature_store_path()`<br> - Selected feature list: `data/training/selected_features.txt`<br> - Feature scores: `data/training/feature_scores.parquet` (falls back to `.csv`)<br> - Trained bundle + metadata: `models/best_model.joblib`, `data/training/feature_metadata.json` |
| **Runtime Contracts** | Calls `property_adviser.predict.model_prediction` for inference and confidence intervals, and uses `property_adviser.predict.feature_store` helpers to hydrate suburb/street pickers. |
| **UX Enhancements** | Shows feature-strength badges (green/orange/blue) based on quartiles of `feature_scores`, and explains confidence intervals using the validation RMSE. |
| **Configuration Touchpoints** | - Feature aliases live in the module-level `FEATURE_ALIASES` map.<br> - Confidence-interval logic inspects the newest `models/model_scores_*.csv` file.<br> - Street/suburb lists are governed by the preprocessing configs that control the derived dataset. |

#### Running Locally

```bash
uv run streamlit run app/predict_app.py
```

Optionally pass Streamlit flags (e.g. `--server.port 8502`). Make sure preprocessing, feature selection,
and training have produced the dependent artefacts; otherwise the app will raise informative
`FileNotFoundError` messages.

### Market Insights Dashboard (`app/market_insights_app.py`)

| Topic | Details |
| --- | --- |
| **Purpose** | Client-facing analytics hub highlighting price drivers, demand shifts, and suburb timelines. |
| **Entry Point** | `uv run streamlit run app/market_insights_app.py` |
| **Upstream Artefacts** | Derived dataset (`feature_store_path()`), feature scores, selected features. |
| **Runtime Contracts** | Uses `property_adviser.predict.feature_store` for suburb-level data and respects `exclude_columns` from `config/features.yml` when presenting driver importance. Charts rendered with Altair. |
| **Sections** | **Market Drivers** (feature importance + driver vs price exploration), **Demand & Growth** (YoY tables, demand heatmap), **Price Timelines** (median price & transaction trendlines). |

#### Running Locally

```bash
uv run streamlit run app/market_insights_app.py
```

Ensure preprocessing and feature-selection artefacts exist; missing inputs will raise instructions to run the upstream pipeline.

### Adding New Apps

1. Place the Streamlit script under `app/` (e.g., `app/new_tool.py`).
2. Document the new app in this file with the same table format.
3. Keep heavy business logic inside the core modules so UI code stays thin and testable.
4. If an app introduces new data dependencies, update the relevant pipeline doc (`PREPROCESS_MODULE.md`, `FEATURE_SELECTION.md`, etc.) so the dependency chain stays clear.

### Maintenance Tips

- Treat `app/` as the presentation layer: defer feature engineering and prediction logic to the
  `property_adviser` packages wherever possible.
- When introducing new inputs, confirm the columns exist in the latest `feature_metadata.json`; missing
  columns will cause inference alignment errors.
- Streamlit caches (e.g., `st.cache_resource`) should wrap deterministic, read-only helpers to avoid stale
  results while iterating.
