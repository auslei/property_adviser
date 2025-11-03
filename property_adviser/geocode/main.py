import os

import pandas as pd
import requests

from property_adviser.core.app_logging import log
from property_adviser.geocode.config import GeocodeConfig

# Your Azure Maps subscription key
AZURE_MAPS_KEY = os.environ.get("AZURE_MAPS_KEY")

if not AZURE_MAPS_KEY:
    raise ValueError("Azure Maps API key not found. Please set the AZURE_MAPS_KEY environment variable.")

def geocode_address(address):
    """Geocode a single address using the Azure Maps API."""
    url = "https://atlas.microsoft.com/search/address/json"
    params = {
        "api-version": "1.0",
        "subscription-key": AZURE_MAPS_KEY,
        "query": address
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        if data["results"]:
            lat = data["results"][0]["position"]["lat"]
            lon = data["results"][0]["position"]["lon"]
            return lat, lon
    except requests.exceptions.RequestException as e:
        log("geocode.error", address=address, error=str(e))
    return None, None

def run_geocoding(config: GeocodeConfig):
    """Main function to read data, geocode addresses, and save the enriched data."""
    log("geocode.start", input_path=str(config.input_path), output_path=str(config.output_path), level=config.level)

    if not config.input_path.exists():
        log("geocode.input_not_found", path=str(config.input_path))
        return

    df = pd.read_parquet(config.input_path)

    required_columns = ["streetAddress", "suburb", "postcode"]
    if not all(col in df.columns for col in required_columns):
        log("geocode.missing_columns", missing=[col for col in required_columns if col not in df.columns])
        return

    if config.level == "full":
        df["address_to_geocode"] = df["streetAddress"] + ", " + df["suburb"] + " " + df["postcode"].astype(str)
    else:
        df["address_to_geocode"] = df["streetAddress"] + ", " + df["suburb"]

    # Create a dataframe with unique addresses
    unique_address_df = df[["streetAddress", "suburb", "postcode", "address_to_geocode"]].drop_duplicates()

    # Geocode unique addresses
    unique_addresses = unique_address_df["address_to_geocode"].unique()
    geocoded_data = {}

    log("geocode.geocoding_start", count=len(unique_addresses))
    for i, address in enumerate(unique_addresses):
        lat, lon = geocode_address(address)
        geocoded_data[address] = {"latitude": lat, "longitude": lon}
        if (i + 1) % 100 == 0:
            log("geocode.progress", processed=i + 1, total=len(unique_addresses))

    log("geocode.complete")

    # Create a dataframe from the geocoded data
    geo_df = pd.DataFrame.from_dict(geocoded_data, orient="index")
    geo_df.index.name = "address_to_geocode"
    
    # Merge the geocoded data with the unique addresses
    output_df = unique_address_df.merge(geo_df, left_on="address_to_geocode", right_index=True)

    # Select and save the required columns
    output_df = output_df[["streetAddress", "suburb", "postcode", "latitude", "longitude"]]
    
    # Save the enriched dataframe
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(config.output_path, index=False)
    log("geocode.saved", path=str(config.output_path))