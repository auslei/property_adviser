## Geocode Module Agent Guide

This guide provides instructions for interacting with the Gemini agent within the `property_adviser/geocode` module.

### Purpose & Scope
The `geocode` module is responsible for enriching the property data with latitude and longitude coordinates. It takes a dataset with address information and produces a new dataset with unique addresses and their corresponding geocodes.

### Gemini's Role
- **Geocoding Logic:** Assist in implementing and modifying the geocoding logic.
- **Output Formatting:** Help define and format the output of the geocoding process.
- **API Integration:** Help integrate with geocoding services like Azure Maps.

### Interaction Guidelines
- **Configuration:** The geocoding process is configured via `config/geocode.yml`, which specifies the input and output files.
- **Output Schema:** The output is a Parquet file containing unique addresses with their `latitude` and `longitude`. Be mindful of this schema when making changes.
