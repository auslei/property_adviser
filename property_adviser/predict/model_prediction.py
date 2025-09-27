import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from property_adviser.config import MODELS_DIR, TRAINING_DIR
from property_adviser.predict.feature_store import fetch_reference_features


def load_trained_model() -> tuple:
    """
    Load the trained model, its metadata and feature information
    """
    model_candidates = [
        MODELS_DIR / "best_model.joblib",
        MODELS_DIR / "best_model.pkl",
    ]
    model_path = next((p for p in model_candidates if p.exists()), model_candidates[0])
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run model training first.")
    
    metadata_path = TRAINING_DIR / "feature_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Feature metadata not found. Run feature selection first.")
    
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        if "model" not in bundle or not hasattr(bundle["model"], "predict"):
            raise ValueError(
                "Model bundle is missing a usable 'model' object. Did training finish successfully?"
            )
        model = bundle["model"]
    else:
        model = bundle
    metadata = json.loads(metadata_path.read_text())
    
    return model, metadata


def _prepare_prediction_data(
    properties: List[Dict[str, Any]], metadata: Dict[str, Any]
) -> pd.DataFrame:
    """Prepare model-ready features from minimal property descriptors."""
    if not properties:
        raise ValueError("At least one property must be provided for prediction")

    feature_metadata = metadata.get("feature_metadata") or {}
    model_columns: List[str] = feature_metadata.get("model_input_columns") or metadata.get("selected_features", [])
    if not model_columns:
        raise ValueError("Feature metadata is missing required model column information")

    numeric_features: List[str] = feature_metadata.get("numeric_features", metadata.get("numeric_features", []))
    categorical_features: List[str] = feature_metadata.get("categorical_features", metadata.get("categorical_features", []))

    impute_numeric: Dict[str, Any] = (feature_metadata.get("impute", {}).get("numeric")
                                      or metadata.get("impute", {}).get("numeric")
                                      or {})
    impute_categorical: Dict[str, Any] = (feature_metadata.get("impute", {}).get("categorical")
                                          or metadata.get("impute", {}).get("categorical")
                                          or {})

    df = pd.DataFrame(properties)

    # Normalise expected column names
    rename_map = {
        "yearmonth": "saleYearMonth",
        "landSize": "landSizeM2",
        "floorSize": "floorSizeM2",
        "yearBuild": "yearBuilt",
        "yearbuild": "yearBuilt",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "suburb" not in df.columns:
        raise ValueError("'suburb' is required for prediction inputs")

    # Ensure saleYearMonth is present and numeric
    if "saleYearMonth" not in df.columns:
        raise ValueError("'saleYearMonth' is required for prediction inputs")
    df["saleYearMonth"] = pd.to_numeric(df["saleYearMonth"], errors="coerce")

    # Fill known numeric identifiers before derivations
    if "yearBuilt" not in df.columns:
        df["yearBuilt"] = np.nan
    if "yearBuilt" in impute_numeric:
        df["yearBuilt"] = df["yearBuilt"].fillna(impute_numeric["yearBuilt"])

    # Derive temporal components
    sale_year = (df["saleYearMonth"] // 100).astype("Int64")
    sale_month = (df["saleYearMonth"] % 100).astype("Int64")
    if "saleYear" in model_columns:
        df["saleYear"] = sale_year
    if "saleMonth" in model_columns:
        df["saleMonth"] = sale_month

    # Property age features
    if {"propertyAge", "propertyAgeBand"}.intersection(model_columns):
        property_age = sale_year.astype("Float64") - pd.to_numeric(df.get("yearBuilt"), errors="coerce")
        property_age = property_age.where(property_age >= 0)
        df["propertyAge"] = property_age

        if "propertyAgeBand" in model_columns:
            age_meta = feature_metadata.get("property_age", {})
            bands = age_meta.get("bands", [5, 20])
            labels = age_meta.get("labels", ["0-5", "6-20", "21+"])
            bins = [-np.inf] + list(bands) + [np.inf]
            df["propertyAgeBand"] = pd.cut(df["propertyAge"], bins=bins, labels=labels)

    # Align optional numeric columns
    if "landSizeM2" in df.columns:
        df["landSizeM2"] = pd.to_numeric(df["landSizeM2"], errors="coerce")
    if "floorSizeM2" in df.columns:
        df["floorSizeM2"] = pd.to_numeric(df["floorSizeM2"], errors="coerce")

    # Build the model-ready frame
    prepared = pd.DataFrame(index=df.index, columns=model_columns)
    for idx, row in df.iterrows():
        for column in model_columns:
            if column in row.index:
                prepared.at[idx, column] = row[column]

        suburb = str(row.get("suburb", "")).strip()
        sale_year_month = pd.to_numeric(row.get("saleYearMonth"), errors="coerce")
        if suburb and not pd.isna(sale_year_month):
            try:
                reference = fetch_reference_features(
                    suburb=suburb,
                    sale_year_month=int(sale_year_month),
                    columns=[col for col in model_columns if col not in {"saleYearMonth"}],
                )
            except FileNotFoundError:
                reference = None

            if reference is not None:
                for column, value in reference.items():
                    if column in prepared.columns and (
                        pd.isna(prepared.at[idx, column]) or column not in row.index
                    ):
                        prepared.at[idx, column] = value

    for column in model_columns:
        if column not in prepared.columns:
            continue
        if prepared[column].isna().all():
            if column in impute_numeric:
                prepared[column] = impute_numeric[column]
            elif column in impute_categorical:
                prepared[column] = impute_categorical[column]

    # Enforce dtypes
    for column in numeric_features:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    for column in categorical_features:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("object")

    # Final imputation fallback
    for column, value in impute_numeric.items():
        if column in prepared.columns:
            prepared[column] = prepared[column].fillna(value)

    for column, value in impute_categorical.items():
        if column in prepared.columns:
            prepared[column] = prepared[column].fillna(value)

    # Ensure categorical bands become plain strings
    if "propertyAgeBand" in prepared.columns:
        prepared["propertyAgeBand"] = prepared["propertyAgeBand"].astype(str).replace({"nan": np.nan})
        prepared["propertyAgeBand"] = prepared["propertyAgeBand"].fillna(impute_categorical.get("propertyAgeBand", "Unknown"))

    return prepared


def predict_property_price(
    yearmonth: int,
    bed: int,
    bath: int,
    car: int,
    property_type: str,
    street: str,
    suburb: str,
    *,
    land_size: Optional[float] = None,
    floor_size: Optional[float] = None,
    year_built: Optional[int] = None,
    model: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    """Predict sale price for a single property using the trained model."""
    if model is None or metadata is None:
        model, metadata = load_trained_model()
    
    properties = [
        {
            "yearmonth": yearmonth,
            "bed": bed,
            "bath": bath,
            "car": car,
            "propertyType": property_type,
            "street": street,
            "suburb": suburb,
            "landSize": land_size,
            "floorSize": floor_size,
            "yearBuild": year_built,
        }
    ]
    input_data = _prepare_prediction_data(properties, metadata)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the predicted price
    return float(prediction[0])


def predict_property_prices_batch(
    properties: List[Dict[str, Any]]
) -> List[float]:
    """
    Predict property prices for multiple properties at once. Each dictionary should
    contain at minimum: `saleYearMonth`, `suburb`, `bed`, `bath`, `car`, `propertyType`, and
    may optionally include `street`, `landSize`, `floorSize`, `yearBuild`.
    """
    model, metadata = load_trained_model()
    
    input_data = _prepare_prediction_data(properties, metadata)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Return list of predicted prices
    return [float(pred) for pred in predictions]


def predict_with_confidence_interval(
    yearmonth: int,
    bed: int,
    bath: int,
    car: int,
    property_type: str,
    street: str,
    suburb: str,
    *,
    land_size: Optional[float] = None,
    floor_size: Optional[float] = None,
    year_built: Optional[int] = None,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Predict property price with confidence interval using model uncertainty.

    This is a simplified approach that uses the Root Mean Squared Error (RMSE)
    from the model's validation set as a measure of uncertainty. A more robust
    approach would involve techniques like bootstrapping or using models that
    natively provide prediction intervals.
    """
    model, metadata = load_trained_model()
    
    # Load validation results to estimate prediction uncertainty
    metrics_path = MODELS_DIR / "model_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        # Take the best model's RMSE as a measure of uncertainty
        best_model_metrics = min(metrics, key=lambda x: x.get('r2', float('-inf')))
        rmse = best_model_metrics.get('rmse', 0)
    else:
        rmse = 0  # Default uncertainty if no metrics available
    
    # Make the primary prediction
    predicted_price = predict_property_price(
        yearmonth,
        bed,
        bath,
        car,
        property_type,
        street,
        suburb,
        land_size=land_size,
        floor_size=floor_size,
        year_built=year_built,
        model=model,
        metadata=metadata,
    )
    
    # Calculate confidence interval (simplified approach)
    z_score = 1.96 if confidence_level == 0.95 else 2.58  # Approximate z-score
    margin_of_error = z_score * rmse
    
    return {
        'predicted_price': predicted_price,
        'lower_bound': predicted_price - margin_of_error,
        'upper_bound': predicted_price + margin_of_error,
        'confidence_level': confidence_level
    }


if __name__ == "__main__":
    # Example usage
    try:
        # Example prediction
        price = predict_property_price(
            yearmonth=202506,  # June 2025
            bed=3,
            bath=2,
            car=2,
            property_type="House",
            street="Example Street"
        )
        print(f"Predicted price: ${price:,.2f}")
        
        # Example with confidence interval
        result = predict_with_confidence_interval(
            yearmonth=202506,
            bed=3,
            bath=2,
            car=2,
            property_type="House", 
            street="Example Street"
        )
        print(f"Predicted price: ${result['predicted_price']:,.2f}")
        print(f"Confidence interval: ${result['lower_bound']:,.2f} - ${result['upper_bound']:,.2f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model has been trained before making predictions.")
