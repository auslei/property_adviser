## Clean Module Agent Guide

This guide provides instructions for interacting with the Gemini agent within the `property_adviser/clean` module.

### Purpose & Scope
The `clean` module is responsible for cleaning and standardizing the raw property data. This includes:
- Handling missing values.
- Standardizing column names and text formats.
- Applying category mappings.
- Filtering out invalid data (e.g., sales with a price of 0).
- Removing outliers.

### Gemini's Role
- **Data Cleaning:** Assist in implementing and configuring data cleaning and validation rules.
- **Configuration:** Help manage the cleaning process via the `config/clean.yml` file.

### Interaction Guidelines
- **Configuration-Driven:** The cleaning process is controlled through the `config/clean.yml` file.
- **`data_cleaning` section:** Use the `data_cleaning` section in `config/clean.yml` to define a pipeline of cleaning steps, such as `drop_na`, `drop_zeros`, and `remove_outliers`.
- **Extensibility:** When proposing new cleaning logic, consider if it can be implemented as a new configurable step in the `data_cleaning` pipeline.
