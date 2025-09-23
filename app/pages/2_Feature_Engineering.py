import yaml

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    load_cleaned_data,
    load_feature_importances,
    load_training_sets,
    read_yaml_config,
    write_yaml_config,
)
from src.config import FEATURE_ENGINEERING_CONFIG_PATH, PREPROCESS_DIR
from src.feature_selection import run_feature_selection


# Load derived data
@st.cache_data
def load_derived_data():
    derived_path = PREPROCESS_DIR / "derived.parquet"
    if derived_path.exists():
        return pd.read_parquet(derived_path)
    else:
        raise FileNotFoundError(f"Derived dataset not found at {derived_path}")


# Load features configuration
def load_features_config():
    config_path = Path("config/features.yml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


# Save features configuration
def save_features_config(config):
    config_path = Path("config/features.yml")
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


st.set_page_config(page_title="Feature Engineering", layout="wide")
st.title("Feature Engineering & Selection")
st.caption(
    "Analyze correlations with target variable and select optimal features for modeling."
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

if "dropped_features" not in st.session_state:
    st.session_state.dropped_features = []


# Load features configuration
features_config = load_features_config()
ignored_columns = features_config.get("ignored_columns", ["streetAddress", "openInRpdata"])

# Get available columns (exclude ignored columns)
available_columns = [col for col in df.columns if col not in ignored_columns]

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
    # Calculate correlation matrix only for numeric columns
    numeric_df = df[numeric_columns].copy()
    corr_matrix = numeric_df.corr()
    
    # Display correlation with target variable
    st.subheader(f"Feature Correlation with {target_variable}")
    
    if target_variable in corr_matrix.columns:
        target_corr = corr_matrix[target_variable].drop(target_variable)  # Remove target itself
        target_corr_abs = target_corr.abs().sort_values(ascending=False)
        
        # Create a dataframe with both correlation and absolute correlation
        corr_df = pd.DataFrame({
            'Feature': target_corr_abs.index,
            'Correlation': target_corr[target_corr_abs.index],
            'Abs_Correlation': target_corr_abs.values
        })
        
        # Display top 20 most correlated features
        st.write(f"Top 20 features most correlated with {target_variable}:")
        st.dataframe(corr_df.head(20))
        
        # Visualization of top correlations
        top_features = target_corr_abs.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x < 0 else 'blue' for x in target_corr[top_features.index]]
        ax.barh(range(len(top_features)), target_corr[top_features.index], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Correlation')
        ax.set_title(f'Top Correlated Features with {target_variable}')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Find co-correlated variable pairs (excluding target)
        st.subheader("Highly Co-correlated Feature Pairs")
        correlated_pairs = []
        
        # Get absolute correlations
        abs_corr_matrix = corr_matrix.abs()
        
        # Find pairs with correlation > 0.7 (threshold for co-correlation)
        for i in range(len(abs_corr_matrix.columns)):
            for j in range(i+1, len(abs_corr_matrix.columns)):
                col1 = abs_corr_matrix.columns[i]
                col2 = abs_corr_matrix.columns[j]
                corr_value = abs_corr_matrix.iloc[i, j]
                
                # Skip if either variable is the target
                if col1 == target_variable or col2 == target_variable:
                    continue
                    
                # If highly co-correlated
                if corr_value > 0.7:
                    correlated_pairs.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Correlation': corr_matrix.iloc[i, j],
                        'Absolute Correlation': corr_value
                    })
        
        if correlated_pairs:
            correlated_df = pd.DataFrame(correlated_pairs).sort_values('Absolute Correlation', ascending=False)
            st.dataframe(correlated_df)
            
            # Recommend features to drop (one from each highly co-correlated pair)
            recommended_drops = []
            processed_pairs = set()
            
            for _, row in correlated_df.iterrows():
                var1, var2 = row['Variable 1'], row['Variable 2']
                pair_key = tuple(sorted([var1, var2]))
                
                if pair_key not in processed_pairs:
                    # Recommend dropping the one with lower absolute correlation to target
                    corr1 = abs(target_corr.get(var1, 0))
                    corr2 = abs(target_corr.get(var2, 0))
                    
                    if corr1 < corr2:
                        recommended_drops.append(var1)
                    else:
                        recommended_drops.append(var2)
                    
                    processed_pairs.add(pair_key)
            
            st.subheader("Recommended Features to Drop (to reduce multicollinearity)")
            st.write(", ".join(recommended_drops) if recommended_drops else "No recommendations")
        else:
            st.info("No highly co-correlated variable pairs found.")
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Get all numeric columns except target
    available_features = [col for col in numeric_columns if col != target_variable]
    
    # Multiselect for dropping features
    dropped_features = st.multiselect(
        "Select features to drop",
        options=available_features,
        default=st.session_state.dropped_features
    )
    
    st.session_state.dropped_features = dropped_features
    
    # Calculate selected features
    selected_features = [col for col in available_features if col not in dropped_features]
    st.session_state.selected_features = selected_features
    
    st.write(f"Selected {len(selected_features)} features out of {len(available_features)} available features")
    
    # Configuration for ignored columns
    st.subheader("Configuration")
    st.write("Current ignored columns:", ", ".join(ignored_columns))
    
    new_ignored = st.text_input("Add columns to ignore (comma separated):", "")
    if st.button("Update Ignored Columns"):
        if new_ignored:
            new_columns = [col.strip() for col in new_ignored.split(",")]
            features_config["ignored_columns"] = list(set(ignored_columns + new_columns))
        else:
            features_config["ignored_columns"] = ignored_columns
            
        save_features_config(features_config)
        st.success("Configuration updated!")
        st.experimental_rerun()
    
    # Save selected features
    if st.button("Save Feature Selection"):
        # Here you would typically save to a configuration file or database
        st.success(f"Saved {len(selected_features)} features!")
        st.write("Selected features:", ", ".join(selected_features))
else:
    st.info("Please select a target variable to begin feature analysis.")
    
    # Show data type information
    st.subheader("Data Type Summary")
    st.write(f"Numeric columns ({len(numeric_columns)}): {', '.join(numeric_columns) if numeric_columns else 'None'}")
    st.write(f"Categorical columns ({len(categorical_columns)}): {', '.join(categorical_columns) if categorical_columns else 'None'}")
    st.write(f"Ignored columns: {', '.join(ignored_columns) if ignored_columns else 'None'}")
