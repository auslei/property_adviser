## Derive Module Agent Guide

This guide provides instructions for interacting with the Gemini agent within the `property_adviser/derive` module.

### Purpose & Scope
The `derive` module is responsible for feature engineering. It takes the cleaned property data and creates a rich set of features for the model, including:
- Time-based features (e.g., cyclical month features).
- Ratio features (e.g., land size per bedroom).
- Rolling and time-aggregated features to capture market trends.
- The primary target variable for the model (`normalized_price`).

### Gemini's Role
- **Feature Engineering:** Assist in designing and implementing new features.
- **Target Variable:** Help define and generate the target variable for the model.
- **Pipeline Management:** Manage the derivation pipelines defined in the `config/derive*.yml` files.

### Interaction Guidelines
- **Configuration-Driven:** The feature derivation process is controlled through YAML files in the `config/` directory (e.g., `derive1.yml`, `derive2.yml`).
- **Step-by-Step:** The derivation process is broken down into a series of steps, each defined in the YAML configuration.
- **Clarity:** When proposing new features, explain the logic and the expected benefit.
