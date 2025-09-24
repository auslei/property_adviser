import yaml

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy.stats import f_oneway, chi2_contingency

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
if "X_preview" not in st.session_state:
    st.session_state.X_preview = None
if "y_preview" not in st.session_state:
    st.session_state.y_preview = None


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
    # Calculate correlation matrix for numeric columns
    numeric_df = df[numeric_columns].copy()
    corr_matrix_numeric = numeric_df.corr()
    
    # Split into tabs for numeric and categorical analysis
    tab1, tab2 = st.tabs(["Numeric Features", "Categorical Features"])
    
    with tab1:
        # Display correlation with target variable for numeric features
        st.subheader(f"Numeric Feature Correlation with {target_variable}")
        
        if target_variable in corr_matrix_numeric.columns:
            target_corr = corr_matrix_numeric[target_variable].drop(target_variable)  # Remove target itself
            target_corr_abs = target_corr.abs().sort_values(ascending=False)
            
            # Create a dataframe with both correlation and absolute correlation
            corr_df = pd.DataFrame({
                'Feature': target_corr_abs.index,
                'Correlation': target_corr[target_corr_abs.index],
                'Abs_Correlation': target_corr_abs.values
            })
            
            # Display top 20 most correlated features
            if not corr_df.empty:
                st.write(f"Top 20 numeric features most correlated with {target_variable}:")
                st.dataframe(corr_df.head(20))
                
                # Visualization of top correlations
                top_features = target_corr_abs.head(15)
                if not top_features.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['red' if x < 0 else 'blue' for x in target_corr[top_features.index]]
                    ax.barh(range(len(top_features)), target_corr[top_features.index], color=colors)
                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(top_features.index)
                    ax.set_xlabel('Correlation')
                    ax.set_title(f'Top Correlated Numeric Features with {target_variable}')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info(f"No numeric features found to correlate with {target_variable}")
        
        # Find co-correlated numeric variable pairs (excluding target)
    with st.expander("View All Co-correlated Numeric Feature Pairs", expanded=False):
        st.subheader("All Co-correlated Numeric Feature Pairs")
        correlated_pairs = []
        
        # Get absolute correlations
        abs_corr_matrix = corr_matrix_numeric.abs()
        
        # Find pairs with correlation > 0.1 (showing all correlations above low threshold)
        for i in range(len(abs_corr_matrix.columns)):
            for j in range(i+1, len(abs_corr_matrix.columns)):
                col1 = abs_corr_matrix.columns[i]
                col2 = abs_corr_matrix.columns[j]
                corr_value = abs_corr_matrix.iloc[i, j]
                
                # Skip if either variable is the target
                if col1 == target_variable or col2 == target_variable:
                    continue
                    
                # Include all pairs with meaningful correlation
                if corr_value > 0.1:
                    correlated_pairs.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Correlation': corr_matrix_numeric.iloc[i, j],
                        'Absolute Correlation': corr_value
                    })
        
        if correlated_pairs:
            correlated_df = pd.DataFrame(correlated_pairs).sort_values('Absolute Correlation', ascending=False)
            st.dataframe(correlated_df)
            
            # Recommend features to drop (one from each highly co-correlated pair)
            recommended_drops = []
            processed_pairs = set()
            
            high_correlation_pairs = [pair for pair in correlated_pairs if pair['Absolute Correlation'] > 0.7]  # High correlation threshold
            
            for pair in high_correlation_pairs:
                var1, var2 = pair['Variable 1'], pair['Variable 2']
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
            
            if recommended_drops:
                st.subheader("Recommended Numeric Features to Drop (to reduce multicollinearity)")
                st.write(", ".join(recommended_drops))
            else:
                st.info("No highly co-correlated pairs (corr > 0.7) found that require dropping.")
        else:
            st.info("No co-correlated numeric variable pairs found.")
    
    # Create anova_df variable in the main scope so it's accessible later
    anova_df = pd.DataFrame()  # Initialize as empty
    
    with tab2:
        # Categorical feature analysis
        st.subheader(f"Categorical Feature Analysis with {target_variable}")
        
        if categorical_columns:
            # Create a method to calculate association between categorical and numeric variables
            # Using ANOVA (F-statistic) to measure the strength of association
            anova_results = []
            
            for cat_col in categorical_columns:
                # Create a copy removing NaN values
                clean_data = df[[cat_col, target_variable]].dropna()
                
                if clean_data.empty or clean_data[cat_col].nunique() < 2:
                    continue
                
                # Use ANOVA to test association between categorical and numeric
                from scipy.stats import f_oneway
                
                # Group the target values by each category
                groups_data = [group[1].values for group in clean_data.groupby(cat_col)[target_variable]]
                
                # Only include groups with at least 2 observations
                groups_data = [group for group in groups_data if len(group) >= 2]
                
                if len(groups_data) >= 2:
                    try:
                        f_stat, p_val = f_oneway(*groups_data)
                        
                        # Effect size: eta squared (for ANOVA)
                        # Calculate total sum of squares
                        total_mean = clean_data[target_variable].mean()
                        total_ss = sum((clean_data[target_variable] - total_mean) ** 2)
                        
                        # Calculate between-group sum of squares
                        group_means = clean_data.groupby(cat_col)[target_variable].mean()
                        group_counts = clean_data.groupby(cat_col)[target_variable].count()
                        between_ss = sum(group_counts * (group_means - total_mean) ** 2)
                        
                        # eta squared as effect size
                        eta_squared = between_ss / total_ss if total_ss != 0 else 0
                        
                        anova_results.append({
                            'Feature': cat_col,
                            'F_Statistic': f_stat,
                            'P_Value': p_val,
                            'Effect_Size': eta_squared,  # Higher means stronger association
                        })
                    except:
                        # If ANOVA fails, skip this column
                        continue
            
            if anova_results:
                anova_df = pd.DataFrame(anova_results)
                anova_df = anova_df.sort_values('Effect_Size', ascending=False)
                
                st.write(f"Categorical features sorted by association strength with {target_variable}:")
                st.dataframe(anova_df)
                
                # Visualization of categorical associations
                if not anova_df.empty:
                    top_cat_features = anova_df.head(10)  # Top 10 categorical features
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(range(len(top_cat_features)), top_cat_features['Effect_Size'])
                    ax.set_yticks(range(len(top_cat_features)))
                    ax.set_yticklabels(top_cat_features['Feature'])
                    ax.set_xlabel('Effect Size (Eta Squared)')
                    ax.set_title(f'Top Categorical Features Associated with {target_variable}')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info(f"No significant categorical features to analyze with {target_variable}")
        else:
            st.info("No categorical features found in the dataset.")
        
        # Categorical-categorical association analysis
        if len(categorical_columns) >= 2:
            with st.expander("Categorical-Categorical Associations (Cramér's V)", expanded=False):
                st.subheader("Categorical-Categorical Associations")
                
                # Calculate Cramér's V for categorical variable pairs
                cramers_v_results = []
                
                for i in range(len(categorical_columns)):
                    for j in range(i+1, len(categorical_columns)):
                        col1, col2 = categorical_columns[i], categorical_columns[j]
                        
                        # Create contingency table
                        contingency_table = pd.crosstab(df[col1].fillna('Missing'), df[col2].fillna('Missing'))
                        
                        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                            continue  # Need at least 2x2 table
                        
                        # Calculate chi-squared
                        try:
                            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                            
                            # Calculate Cramér's V
                            n = contingency_table.sum().sum()
                            min_dim = min(contingency_table.shape) - 1
                            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                            
                            cramers_v_results.append({
                                'Variable 1': col1,
                                'Variable 2': col2,
                                'Cramér\'s V': cramers_v,
                                'Chi2': chi2,
                                'P_Value': p_value
                            })
                        except:
                            # If calculation fails, skip this pair
                            continue
                
                if cramers_v_results:
                    cramers_v_df = pd.DataFrame(cramers_v_results)
                    cramers_v_df = cramers_v_df.sort_values('Cramér\'s V', ascending=False)
                    
                    # Filter to only show meaningful associations (threshold > 0.1)
                    meaningful_associations = cramers_v_df[cramers_v_df["Cramér's V"] > 0.1]
                    
                    if not meaningful_associations.empty:
                        st.write(f"Categorical variable pairs with meaningful associations (Cramér's V > 0.1):")
                        st.dataframe(meaningful_associations)
                        
                        # Show top categorical associations visualization
                        top_cat_pairs = meaningful_associations.head(10)
                        if not top_cat_pairs.empty:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(range(len(top_cat_pairs)), top_cat_pairs["Cramér's V"])
                            ax.set_yticks(range(len(top_cat_pairs)))
                            ax.set_yticklabels([f"{row['Variable 1']} - {row['Variable 2']}" for _, row in top_cat_pairs.iterrows()])
                            ax.set_xlabel("Cramér's V")
                            ax.set_title("Top Categorical-Categorical Associations")
                            ax.grid(axis='x', alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.info("No meaningful categorical-categorical associations found (Cramér's V > 0.1).")
                        
                    # Add all associations to the recommended features if they're not already there  
                    for _, row in meaningful_associations.iterrows():
                        # Add variables to recommended features if they have strong categorical-categorical association
                        var1, var2 = row['Variable 1'], row['Variable 2']
                        # We don't add these directly to recommendations, just show them as relevant
                        
                else:
                    st.info("No categorical-categorical associations found.")
        else:
            st.info("Need at least 2 categorical variables to analyze categorical-categorical associations.")
    
    # Combined feature selection (include both numeric and categorical)
    all_available_features = numeric_columns + categorical_columns
    if target_variable in all_available_features:
        all_available_features.remove(target_variable)
    
    # Calculate recommended features based on correlation/association strength
    recommended_features = []
    
    # Add highly correlated numeric features
    if target_variable in corr_matrix_numeric.columns:
        target_corr = corr_matrix_numeric[target_variable].drop(target_variable)
        target_corr_abs = target_corr.abs().sort_values(ascending=False)
        
        # Select numeric features with correlation > 0.1 (adjustable threshold)
        for feature, corr_val in target_corr_abs.items():
            if abs(corr_val) > 0.1:  # Only include features with meaningful correlation
                recommended_features.append({
                    'Feature': feature,
                    'Score': abs(corr_val),
                    'Type': 'Numeric',
                    'Direction': 'positive' if corr_val >= 0 else 'negative'
                })
    
    # Add categorical features with strong associations
    if not anova_df.empty:
        for _, row in anova_df.iterrows():
            if row['Effect_Size'] > 0.01:  # Only include features with meaningful association
                recommended_features.append({
                    'Feature': row['Feature'],
                    'Score': row['Effect_Size'],
                    'Type': 'Categorical',
                    'Direction': 'association'
                })
    
    # Sort recommended features by score
    recommended_features = sorted(recommended_features, key=lambda x: x['Score'], reverse=True)
    
    if recommended_features:
        # Display recommendations table
        recommended_df = pd.DataFrame(recommended_features)
        st.write(f"Top recommended features (based on correlation/association with {target_variable}):")
        st.dataframe(recommended_df[['Feature', 'Score', 'Type']].head(20))
        
        # Create a default selection based on recommendations
        default_features = [f['Feature'] for f in recommended_features[:15] if f['Feature'] in all_available_features]
    else:
        # Fallback to all available features if no strong correlations found
        default_features = all_available_features[:20]  # Limit to first 20 features
    
    # Auto Select button functionality
    col_auto, col_info = st.columns([1, 3])
    with col_auto:
        if st.button("🔄 Auto Select Features"):
            # Auto-select features based on recommendations
            auto_selected = [f['Feature'] for f in recommended_features if f['Feature'] in all_available_features and f['Score'] > 0.15]
            if not auto_selected:  # If no features meet the threshold, select top 10
                auto_selected = [f['Feature'] for f in recommended_features[:10] if f['Feature'] in all_available_features]
            st.session_state.selected_features = auto_selected if auto_selected else default_features
            st.rerun()
    
    with col_info:
        st.info(f"Recommended: {len([f for f in recommended_features if f['Score'] > 0.15])} features with score > 0.15")
    
    # Allow user to add or remove features
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_features = st.multiselect(
            "Select features to use for modeling:",
            options=sorted(all_available_features),
            default=st.session_state.get('selected_features', default_features),
            key='feature_selector'
        )
    
    with col2:
        st.write(f"Selected {len(selected_features)} features")
        st.write(" ")
        if st.button("Use All Features"):
            st.session_state.selected_features = all_available_features
            st.rerun()
    
    # Generate training datasets button
    if st.button("💾 Generate Training Datasets (X, y)", type="primary"):
        if selected_features:
            with st.spinner("Generating training datasets..."):
                try:
                    # Import required modules from feature selection
                    from src.feature_selection import run_feature_selection
                    import json
                    
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
                    import pandas as pd
                    from src.config import TRAINING_DIR
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
