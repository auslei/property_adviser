import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, TRAINING_DIR


def load_trained_model() -> tuple:
    """
    Load the trained model, its metadata and feature information
    """
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run model training first.")
    
    metadata_path = TRAINING_DIR / "feature_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Feature metadata not found. Run feature selection first.")
    
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    
    return model, metadata


def _prepare_prediction_data(
    properties: List[Dict[str, Any]], metadata: Dict[str, Any]
) -> pd.DataFrame:
    """Prepares a DataFrame for prediction from a list of property dictionaries."""
    input_data = pd.DataFrame(properties)

    # Ensure we have the right columns based on training metadata
    selected_features = metadata.get("selected_features", [])

    # Add missing columns with default values if needed
    for feature in selected_features:
        if feature not in input_data.columns:
            if feature in ["saleYear", "saleMonth"]:
                # If we have yearmonth, we can derive these
                if "yearmonth" in input_data.columns:
                    input_data["saleYear"] = input_data["yearmonth"] // 100
                    input_data["saleMonth"] = input_data["yearmonth"] % 100
                else:
                    input_data[feature] = 0  # Default value
            elif feature in metadata.get("numeric_features", []):
                input_data[feature] = 0  # Default numeric value
            elif feature in metadata.get("categorical_features", []):
                input_data[feature] = "Unknown"  # Default categorical value
            else:
                input_data[feature] = 0  # Default fallback

    # Filter to only include features used in training
    return input_data[selected_features]


def predict_property_price(
    yearmonth: int,
    bed: int,
    bath: int, 
    car: int,
    property_type: str,
    street: str,
    model: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> float:
    """
    Predict property price based on the given features using the trained model
    """
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
    Predict property prices for multiple properties at once
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
        yearmonth, bed, bath, car, property_type, street, model, metadata
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
