
## Macro Module Agent Guide

This guide provides instructions for interacting with the Gemini agent within the `property_adviser/macro` module.

### Purpose & Scope
The `macro` module is responsible for fetching and processing macroeconomic data.

### Gemini's Role
- **Data Ingestion:** Assist in adding new data sources for macroeconomic indicators.
- **Data Processing:** Help in cleaning and transforming the macroeconomic data.
- **Integration:** Integrate the macroeconomic data with the main property dataset.

### Interaction Guidelines
- **Specify Data Sources:** When requesting to add new data sources, provide the URL and a description of the data.
- **Configuration:** Use the `config/macro.yml` file to configure the data sources and processing steps.
- **Output Schema:** Ensure that the output files have a consistent schema.
