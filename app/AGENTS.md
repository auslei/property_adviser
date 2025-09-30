# Application Layer Agent Guide

## Purpose & Scope
- Host Streamlit applications that visualise model outputs, inspect feature drivers, and support analyst workflows.
- Treat `app/` as the presentation layer; delegate business logic to `property_adviser` packages to preserve cohesion and reuse.

## Design Commitments
- **Clear interface**: Each app exposes a Streamlit script with well-defined inputs and relies on documented services in `property_adviser.predict` and `property_adviser.feature`.
- **High cohesion**: UI composition, layout, and user interaction stay inside `app/`. Data preparation and inference remain in dedicated modules.
- **Low coupling**: Share state through persisted artefacts or `core` helpers—avoid importing sibling apps directly.
- **Reusability**: Common widgets or utilities should live in `core` (or a shared `app/utils.py` if UI-specific) to avoid duplication.

## Included Apps

### Property Price Predictor (`app/predict_app.py`)
- Purpose: interactive UI for estimating sale prices with optional confidence intervals.
- Entry point: `uv run streamlit run app/predict_app.py`
- Dependencies: `models/best_model.joblib`, `data/training/feature_metadata.json`, `data/training/feature_scores.parquet`, `data/training/selected_features.txt`, derived dataset via `property_adviser.predict.feature_store`.
- Runtime contracts: calls `property_adviser.predict.model_prediction` APIs; honours `exclude_columns` from feature selection when surfacing driver badges.

### Market Insights Dashboard (`app/market_insights_app.py`)
- Purpose: analyst-facing dashboard for price drivers, demand signals, and suburb trends.
- Entry point: `uv run streamlit run app/market_insights_app.py`
- Dependencies: derived dataset and feature scores via `feature_store`, selected features list, training artefacts for context.
- Sections: Market Drivers, Demand & Growth, Price Timelines.

## Running Locally
```bash
uv run streamlit run app/predict_app.py
uv run streamlit run app/market_insights_app.py
```
Ensure preprocessing, feature selection, and training stages have produced the referenced artefacts to avoid runtime errors.

## Maintenance Checklist
1. Keep heavy computation in `property_adviser` modules; reference them from Streamlit callbacks.
2. Document new apps here using the same structure and update the root `AGENTS.md` if dependencies change.
3. When introducing new data requirements, confirm preprocessing/feature selection outputs supply them and update their `AGENTS.md` files accordingly.
