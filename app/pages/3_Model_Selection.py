import streamlit as st
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path

from utils import (
    load_training_sets,
    load_model_metrics,
    read_yaml_config,
    write_yaml_config,
)
from src.config import MODEL_CONFIG_PATH, MODELS_DIR, TRAINING_DIR
from src.model_training import train_models


st.set_page_config(page_title="Model Selection", layout="wide")
st.title("Model Selection & Hyperparameter Tuning")
st.caption(
    "Compare different models, tune hyperparameters, and select the best performing model for property price prediction."
)


# Load training data and model metrics
try:
    X, y, feature_metadata = load_training_sets()
    st.success(f"Training data loaded: {X.shape[0]:,} rows × {X.shape[1]} features")
    
    # Show target information
    st.write(f"Target variable: {feature_metadata['target']}")
    st.write(f"Selected features: {len(feature_metadata['selected_features'])} total")
    
    # Display selected features
    with st.expander("Selected Features List", expanded=False):
        st.write("Features selected from Feature Engineering:")
        selected_features_text = ", ".join(feature_metadata['selected_features'])
        st.text_area("Features", value=selected_features_text, height=150)
    
    # Highlight key features for property price prediction
    key_features = ['yearmonth', 'bed', 'bath', 'car', 'propertyType', 'street', 'saleYear', 'saleMonth']
    available_key_features = [f for f in key_features if f in feature_metadata['selected_features']]
    
    st.info(f"Key features for property price prediction found: {', '.join(available_key_features) if available_key_features else 'None found'}")
    
    # Display sample of training data
    with st.expander("Preview Training Data (X, y)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Feature Matrix (X) - Shape: {X.shape}")
            st.dataframe(X.head(10))  # Show first 10 rows
            
        with col2:
            st.write(f"Target Vector (y) - Shape: {y.shape}")
            st.dataframe(y.head(10).to_frame())  # Show first 10 rows as dataframe
    
except FileNotFoundError:
    st.warning("Training data not found. Run preprocessing and feature selection first.")
    st.stop()


# Load model configuration
try:
    model_config = read_yaml_config(MODEL_CONFIG_PATH)
except FileNotFoundError:
    st.warning(f"Model configuration not found at {MODEL_CONFIG_PATH}")
    model_config = {}


# Training controls
st.subheader("Model Configuration")
col1, col2 = st.columns(2)

with col1:
    # Split configuration
    st.write("**Train/Validation Split**")
    test_size = st.slider(
        "Test size", 
        min_value=0.1, 
        max_value=0.5, 
        value=model_config.get("split", {}).get("test_size", 0.2),
        step=0.05
    )
    random_state = st.number_input(
        "Random state", 
        value=model_config.get("split", {}).get("random_state", 42),
        min_value=0,
        max_value=10000
    )

with col2:
    # Model selection
    st.write("**Model Selection**")
    models_config = model_config.get("models", {})
    available_models = ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor", "Lasso", "ElasticNet"]
    
    # Create checkboxes for each model
    selected_models = {}
    for model_name in available_models:
        default_enabled = models_config.get(model_name, {}).get("enabled", True)
        selected_models[model_name] = st.checkbox(
            model_name, 
            value=default_enabled,
            key=f"model_{model_name}"
        )


# Model hyperparameter tuning
st.subheader("Hyperparameter Tuning")

# Create tabs for each model
model_tabs = st.tabs(list(selected_models.keys()))

for i, (model_name, is_selected) in enumerate(selected_models.items()):
    with model_tabs[i]:
        if is_selected:
            st.write(f"Configure hyperparameters for {model_name}")
            
            # Default grid based on model type
            default_grids = {
                "LinearRegression": {},
                "RandomForestRegressor": {
                    "n_estimators": [100, 200, 400],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "GradientBoostingRegressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0]
                },
                "Lasso": {
                    "alpha": [0.1, 1.0, 10.0, 100.0]
                },
                "ElasticNet": {
                    "alpha": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.9]
                }
            }
            
            current_grid = models_config.get(model_name, {}).get("grid", {})
            default_grid = default_grids.get(model_name, {})
            
            # Allow user to customize parameters
            grid_params = {}
            if model_name == "LinearRegression":
                # LinearRegression has limited hyperparameters that can be meaningfully tuned
                st.info(f"{model_name} typically requires minimal hyperparameter tuning")
                # Ensure no parameters are configured for LinearRegression to avoid invalid parameter errors
            elif model_name in default_grids and default_grids[model_name]:
                st.write("**Hyperparameter Grid**")
                
                for param, values in default_grid.items():
                    # Create a form for each parameter
                    st.write(f"`{param}`")
                    param_values = st.text_input(
                        f"Values for {param} (comma-separated)",
                        value=", ".join(map(str, current_grid.get(f"model__{param}", values))),
                        key=f"{model_name}_{param}_input"  # Unique key for each text input
                    )
                    
                    try:
                        # Parse the values
                        parsed_values = [parse_value(v.strip()) for v in param_values.split(",") if v.strip()]
                        grid_params[param] = parsed_values
                    except:
                        st.warning(f"Invalid values for {param}, using defaults")
                        grid_params[param] = values
            else:
                st.info(f"No default hyperparameters for {model_name}, using defaults")
                grid_params = {}
        
        else:
            st.info(f"Model {model_name} is not selected")


def parse_value(value_str):
    """Parse string value to appropriate type"""
    # Try to convert to int
    try:
        if '.' not in value_str:
            return int(value_str)
    except ValueError:
        pass
    
    # Try to convert to float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string if it's not a special keyword
    if value_str.lower() == 'none':
        return None
    return value_str


# Final configuration to save
final_config = {
    "split": {
        "test_size": test_size,
        "random_state": random_state
    },
    "models": {
        name: {
            "enabled": is_selected,
            # For LinearRegression, always use empty grid to avoid invalid parameter errors
            "grid": {} if name == "LinearRegression" else 
                    {f"model__{k}": v for k, v in grid_params.items()} if is_selected and name in selected_models and grid_params else {}
        }
        for name, is_selected in selected_models.items()
    }
    # Manual feature adjustments section removed as requested
}


# Train models button
if st.button("🚀 Train Models", type="primary"):
    with st.spinner("Training models... This may take a few minutes."):
        try:
            # Debug: Show the configuration being used
            st.info(f"Training with configuration: {final_config}")
            
            # Save configuration
            write_yaml_config(MODEL_CONFIG_PATH, final_config)
            
            # Train the models
            result = train_models(final_config)
            
            st.success(f"Model training completed! Best model: {result['best_model']}")
            
            # Show training results
            if 'model_metrics' in st.session_state:
                del st.session_state['model_metrics']  # Clear cache to refresh
            
        except Exception as e:
            import traceback
            st.error(f"Error during model training: {str(e)}")
            st.error(f"Full traceback:\n{traceback.format_exc()}")


# Display model metrics if available
try:
    metrics_df = load_model_metrics()
    
    st.subheader("Model Performance Comparison")
    st.dataframe(metrics_df.style.highlight_max(subset=['r2'], color='lightgreen').highlight_min(subset=['mae', 'rmse'], color='lightgreen'))
    
    # Show best model details
    best_model_idx = metrics_df['r2'].idxmax()
    best_model_row = metrics_df.iloc[best_model_idx]
    
    st.subheader("Best Performing Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", best_model_row['model'])
    col2.metric("R² Score", f"{best_model_row['r2']:.4f}")
    col3.metric("MAE", f"${best_model_row['mae']:,.2f}")
    col4.metric("RMSE", f"${best_model_row['rmse']:,.2f}")
    
    if 'best_params' in best_model_row and pd.notna(best_model_row['best_params']):
        st.write("**Best Hyperparameters:**")
        st.json(best_model_row['best_params'])
    
except FileNotFoundError:
    st.info("No model metrics available. Train models to see performance comparison.")