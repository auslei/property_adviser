import yaml

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import json
import shutil

from property_adviser.common.app_utils import load_training_sets
from property_adviser.config import FEATURE_ENGINEERING_CONFIG_PATH, PREPROCESS_DIR, TRAINING_DIR
from property_adviser.feature_selection_util import run_feature_selection


# Load derived data
@st.cache_data
def load_derived_data():
    derived_path = PREPROCESS_DIR / "derived.parquet"
    if derived_path.exists():
        return pd.read_parquet(derived_path)
    else:
        raise FileNotFoundError(f"Derived dataset not found at {derived_path}")

# Load features configuration - loads the consolidated config
def load_features_config():
    config_path = Path("config/features.yml")  # This is now the consolidated config
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


# Save features configuration - updates exclude columns in consolidated config
def save_features_config(config):
    config_path = Path("config/features.yml")  # Use the consolidated config
    # Load the existing config to preserve all settings
    existing_config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            existing_config = yaml.safe_load(f) or {}
    
    # Update only the exclude_columns part with the new values
    existing_config['exclude_columns'] = config.get('exclude_columns', [])
    
    # Save the updated config
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(existing_config, f)


# Get feature recommendations using the run_feature_selection function
def get_feature_recommendations(df: pd.DataFrame, target_variable: str) -> List[Dict[str, Any]]:
    """Get feature recommendations using the core feature selection module."""
    try:
        # Run feature selection to get feature importances
        feature_info = run_feature_selection({
            "target": target_variable,
            "baseline": {
                "derive_factor": True,  # Use price factor calculation
                "base_target": target_variable,
                "baseline_column": "baselineMedian",
                "transactions_column": "baselineTransactions"
            },
            "correlation_threshold": 0.9,
            "exclude_columns": [],  # Don't exclude anything for recommendations
            "force_keep": []  # Don't force keep anything for recommendations
        })
        
        # Return the feature importances as recommendations
        return feature_info.get('feature_importances', [])
    except Exception as e:
        st.warning(f"Could not compute feature recommendations: {str(e)}")
        return []


st.set_page_config(page_title="Feature Engineering", layout="wide")

# Define callback to update session state when multiselect changes
def update_selected_features():
    # Use the current multiselect key based on version
    multiselect_key = f"feature_selector_{st.session_state.multiselect_version}"
    if multiselect_key in st.session_state:
        st.session_state.selected_features = st.session_state[multiselect_key]

st.title("Feature Engineering & Selection")
st.caption(
    "Analyze feature importances and select optimal features for modeling."
)


try:
    df = load_derived_data()
    st.success("Derived data loaded successfully!")
except FileNotFoundError as e:
    st.error(f"Error loading derived data: {e}")
    st.stop()

# Initialize session state
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []
if "X_preview" not in st.session_state:
    st.session_state.X_preview = None
if "y_preview" not in st.session_state:
    st.session_state.y_preview = None
if "multiselect_version" not in st.session_state:
    st.session_state.multiselect_version = 0


# Load features configuration
features_config = load_features_config()
exclude_columns = features_config.get("exclude_columns", ["streetAddress", "openInRpdata"])

# Get available columns (exclude columns)
available_columns = [col for col in df.columns if col not in exclude_columns]

# Separate column types
numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col in available_columns]
categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns.tolist() if col in available_columns]

# Target variable selection - default to salePrice if available
default_target = "salePrice" if "salePrice" in numeric_columns else (numeric_columns[0] if numeric_columns else None)
target_variable = st.selectbox(
    "Select target variable",
    options=numeric_columns,
    index=numeric_columns.index(default_target) if default_target and default_target in numeric_columns else 0,
    placeholder="Choose a target variable..."
)

if target_variable:
    # Combined feature selection (include both numeric and categorical)
    all_available_features = numeric_columns + categorical_columns
    if target_variable in all_available_features:
        all_available_features.remove(target_variable)
    
    # Use the run_feature_selection function to get feature recommendations
    # Load features configuration to get exclude columns
    features_config = load_features_config()
    exclude_columns = features_config.get("exclude_columns", ["streetAddress", "openInRpdata"])
    
    # Get feature recommendations
    recommended_features = get_feature_recommendations(df, target_variable)
    
    # Display recommendations table
    if recommended_features:
        recommended_df = pd.DataFrame(recommended_features)
        st.write(f"Top recommended features (based on feature importance for predicting {target_variable}):")
        st.dataframe(recommended_df[['feature', 'importance']].head(20))
        
        # Create a default selection based on recommendations (top 15)
        default_features = [f['feature'] for f in recommended_features[:15] if f['feature'] in all_available_features]
    else:
        # Fallback to all available features if no recommendations found
        default_features = all_available_features[:20]  # Limit to first 20 features
    
    # Auto Select button functionality
    col_auto, col_info = st.columns([1, 3])
    with col_auto:
        if st.button("🔄 Auto Select Features"):
            # Auto-select features based on importance scores from run_feature_selection
            # Use a threshold of 0.01 for feature importance (this is the threshold used in the module)
            auto_selected = [f['feature'] for f in recommended_features if f['feature'] in all_available_features and f['importance'] > 0.01]
            if not auto_selected:  # If no features meet the threshold, select top 10
                auto_selected = [f['feature'] for f in recommended_features[:10] if f['feature'] in all_available_features]
            st.session_state.selected_features = auto_selected if auto_selected else default_features
            # Increment version to force multiselect to refresh
            st.session_state.multiselect_version += 1
            st.rerun()
    
    with col_info:
        st.info(f"Recommended: {len([f for f in recommended_features if f['importance'] > 0.01])} features with importance > 0.01")
    
    # Allow user to add or remove features
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use a dynamic key to force refresh when version changes
        multiselect_key = f"feature_selector_{st.session_state.multiselect_version}"
        # The multiselect will use the session state as its default value
        # which should be properly updated by the buttons with st.rerun()
        selected_features = st.multiselect(
            "Select features to use for modeling:",
            options=sorted(all_available_features),
            default=st.session_state.get('selected_features', default_features),
            key=multiselect_key,
            on_change=update_selected_features
        )
        # Update session state to match the current selection
        if multiselect_key in st.session_state:
            st.session_state.selected_features = st.session_state[multiselect_key]
    
    with col2:
        st.write(f"Selected {len(selected_features)} features")
        st.write(" ")
        if st.button("Use All Features"):
            st.session_state.selected_features = all_available_features
            # Increment version to force multiselect to refresh
            st.session_state.multiselect_version += 1
            st.rerun()
    
    # Generate training datasets button
    if st.button("💾 Generate Training Datasets (X, y)", type="primary"):
        if selected_features:
            with st.spinner("Generating training datasets..."):
                try:
                    # Create a temporary config for feature selection
                    # Important: Keep baseline key columns (saleYear, saleMonth, suburb) for baseline calculation
                    baseline_key_cols = {"saleYear", "saleMonth", "suburb"}
                    features_to_exclude = [col for col in all_available_features 
                                         if col not in selected_features and col not in baseline_key_cols]
                    
                    temp_config = {
                        "target": target_variable,
                        "baseline": {
                            "derive_factor": False,  # Use actual prices, not factors
                            "base_target": target_variable,
                            "baseline_column": "baselineMedian",
                            "transactions_column": "baselineTransactions"
                        },
                        "correlation_threshold": 0.9,
                        "exclude_columns": features_to_exclude,
                        "force_keep": selected_features  # Force keep selected features
                    }
                    
                    # Run feature selection with the selected features
                    feature_info = run_feature_selection(temp_config)
                    
                    # Load the generated X and y data for preview
                    X_path = TRAINING_DIR / "X.parquet"
                    y_path = TRAINING_DIR / "y.parquet"
                    
                    if X_path.exists() and y_path.exists():
                        X_preview = pd.read_parquet(X_path)
                        y_preview = pd.read_parquet(y_path)
                        
                        # Store in session state for preview
                        st.session_state.X_preview = X_preview
                        st.session_state.y_preview = y_preview
                        
                        st.success(f"Training datasets generated successfully!")
                        st.write(f"- Features selected: {len(selected_features)}")
                        st.write(f"- Target variable: {target_variable}")
                        st.write(f"- Training samples: {feature_info.get('rows', 'Unknown')}")
                        
                        # Show selected features
                        st.write("**Final selected features:**")
                        st.write(", ".join(feature_info.get("selected_features", [])))
                        
                        # Show preview section
                        with st.expander("Preview Generated Training Data (X, y)", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Feature Matrix (X) - Shape: {X_preview.shape}**")
                                st.dataframe(X_preview.head(10))  # Show first 10 rows
                                
                            with col2:
                                st.write(f"**Target Vector (y) - Shape: {y_preview.shape}**")
                                st.dataframe(y_preview.head(10))  # Show first 10 rows
                    
                    else:
                        st.error("Generated datasets not found after running feature selection.")
                        
                except Exception as e:
                    st.error(f"Error generating training datasets: {str(e)}")
        else:
            st.warning("Please select at least one feature before generating datasets.")
        
    # Data type information at the bottom
    st.subheader("Data Type Summary")
    st.write(f"Numeric columns ({len(numeric_columns)}): {', '.join(numeric_columns) if numeric_columns else 'None'}")
    st.write(f"Categorical columns ({len(categorical_columns)}): {', '.join(categorical_columns) if categorical_columns else 'None'}")
    st.write(f"Excluded columns: {', '.join(exclude_columns) if exclude_columns else 'None'}")
else:
    st.info("Please select a target variable to begin feature analysis.")
    
    # Show data type information
    st.subheader("Data Type Summary")
    st.write(f"Numeric columns ({len(numeric_columns)}): {', '.join(numeric_columns) if numeric_columns else 'None'}")
    st.write(f"Categorical columns ({len(categorical_columns)}): {', '.join(categorical_columns) if categorical_columns else 'None'}")
    st.write(f"Excluded columns: {', '.join(exclude_columns) if exclude_columns else 'None'}")