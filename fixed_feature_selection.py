import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from .config import (
    FEATURE_ENGINEERING_CONFIG_PATH,
    PREPROCESS_DIR,
    RANDOM_STATE,
    TRAINING_DIR,
)
from .configuration import load_yaml
from .suburb_median import (
    GLOBAL_SUBURB_KEY,
    HISTORY_FILENAME,
    load_baseline_median_history,
)

TIME_FEATURES = {"saleDate"}


def _load_feature_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is not None:
        return config
    from .config import FEATURE_ENGINEERING_CONFIG_PATH
    from .configuration import load_yaml
    return load_yaml(FEATURE_ENGINEERING_CONFIG_PATH)


def _load_clean_data() -> pd.DataFrame:
    data_path = PREPROCESS_DIR / "cleaned.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            "Preprocessed data not found. Run src/preprocess.py first."
        )
    return pd.read_parquet(data_path)


def _drop_low_variance(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    columns_to_drop = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].nunique(dropna=False) <= 1:
            columns_to_drop.append(col)
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return df


def _remove_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    drop_candidates = [
        "openInRpdata",
        "parcelDetails",
        "streetAddress",
    ]
    existing = [col for col in drop_candidates if col in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df


def _attach_baseline_median(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Attaches baseline medians to the dataframe for price factor calculation.

    This simplified approach uses only observed historical medians instead of
    the complex forecasting model that was previously used.

    It first attempts to join suburb-specific monthly medians. If a property's
    suburb-month combination doesn't have a historical median, it falls back
    to the global monthly median.
    """
    history = load_baseline_median_history()
    key_cols = ["suburb", "saleYear", "saleMonth"]

    # Ensure proper data types for key columns
    for col in key_cols:
        if col == "suburb":
            if col in history.columns:
                history[col] = history[col].fillna("Unknown").astype(str)
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        else:
            if col in history.columns:
                history[col] = pd.to_numeric(history[col], errors="coerce").astype("Int64")
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].round().astype("Int64")

    df = df.dropna(subset=[col for col in key_cols if col != "suburb"])

    # Create suburb-specific medians with renamed columns
    suburb_history = (
        history[history["suburb"] != GLOBAL_SUBURB_KEY][
            key_cols + ["medianPrice", "transactionCount"]
        ]
        .drop_duplicates(key_cols, keep="last")
        .rename(
            columns={
                "medianPrice": "baselineMedian",
                "transactionCount": "baselineTransactions",
            }
        )
    )

    # Merge suburb medians with main dataframe
    merged = df.merge(suburb_history, on=key_cols, how="left")

    # Handle missing suburb medians by falling back to global medians
    missing_mask = merged["baselineMedian"].isna()
    if missing_mask.any():
        global_history = (
            history[history["suburb"] == GLOBAL_SUBURB_KEY][
                ["saleYear", "saleMonth", "medianPrice", "transactionCount"]
            ]
            .drop_duplicates(["saleYear", "saleMonth"], keep="last")
            .rename(
                columns={
                    "medianPrice": "baselineMedian_global",
                    "transactionCount": "baselineTransactions_global",
                }
            )
        )
        merged = merged.merge(
            global_history,
            on=["saleYear", "saleMonth"],
            how="left",
        )
        merged.loc[missing_mask, "baselineMedian"] = merged.loc[
            missing_mask, "baselineMedian_global"
        ]
        merged.loc[missing_mask, "baselineTransactions"] = merged.loc[
            missing_mask, "baselineTransactions_global"
        ]
        merged = merged.drop(columns=["baselineMedian_global", "baselineTransactions_global"])

    # Fill any remaining missing transaction counts with 0
    merged["baselineTransactions"] = merged["baselineTransactions"].fillna(0)

    return merged, "baselineMedian", "baselineTransactions"


def _correlated_features(df: pd.DataFrame, numeric_columns: List[str], threshold: float = 0.9) -> List[str]:
    if not numeric_columns:
        return []
    corr_matrix = df[numeric_columns].corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper = corr_matrix.where(mask)
    correlated = [
        column
        for column in upper.columns
        if any(upper[column].fillna(0) > threshold)
    ]
    return correlated


def run_feature_selection(config: Optional[Dict[str, Any]] = None) -> Dict[str, object]:
    resolved_config = _load_feature_config(config)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_clean_data()
    if "salePrice" not in df.columns:
        raise ValueError("Expected 'salePrice' column to be present in preprocessed data.")

    target_col = resolved_config.get("target", "priceFactor")
    baseline_cfg = resolved_config.get("baseline", {})
    derive_factor = baseline_cfg.get("derive_factor", True)
    base_target_col = baseline_cfg.get("base_target", "salePrice")
    baseline_col_name = baseline_cfg.get("baseline_column", "baselineMedian")
    transactions_col_name = baseline_cfg.get("transactions_column", "baselineTransactions")

    df = _remove_identifiers(df)
    drop_time_cols = [col for col in TIME_FEATURES if col in df.columns]
    if drop_time_cols:
        df = df.drop(columns=drop_time_cols)

    # Load default exclusions (these were previously in EXCLUDE_COLUMNS)
    default_exclusions = ["streetAddress", "openInRpdata", "parcelDetails"]
    configured_exclusions = [
        col for col in default_exclusions if col in df.columns and col != target_col
    ]
    if configured_exclusions:
        df = df.drop(columns=configured_exclusions)

    manual_existing: List[str] = []
    # Support both old 'drop_columns' and new 'exclude_columns' for backward compatibility
    manual_drop = resolved_config.get("drop_columns", []) + resolved_config.get("exclude_columns", [])
    if manual_drop:
        manual_existing = [col for col in manual_drop if col in df.columns]
        if manual_existing:
            df = df.drop(columns=manual_existing)

    if base_target_col in df.columns:
        df = df[df[base_target_col].notna()]

    df, baseline_col, baseline_tx_col = _attach_baseline_median(df)
    if baseline_col_name != baseline_col and baseline_col in df.columns:
        df = df.rename(columns={baseline_col: baseline_col_name})
        baseline_col = baseline_col_name
    if (
        transactions_col_name
        and baseline_tx_col
        and transactions_col_name != baseline_tx_col
        and baseline_tx_col in df.columns
    ):
        df = df.rename(columns={baseline_tx_col: transactions_col_name})
        baseline_tx_col = transactions_col_name

    df = df[df[baseline_col].notna()]
    df = df[df[baseline_col] > 0]

    if derive_factor:
        numerator_col = base_target_col if base_target_col in df.columns else target_col
        # Check if priceFactor already exists (calculated in preprocessing)
        if target_col in df.columns and target_col != numerator_col:
            # priceFactor already exists, just ensure it's clean
            df = df.replace({target_col: {np.inf: np.nan, -np.inf: np.nan}})
            df = df[df[target_col].notna()]
            df = df[df[target_col] > 0]
        else:
            # Calculate priceFactor as before
            if numerator_col not in df.columns:
                raise ValueError(
                    f"Configured base target column '{numerator_col}' is missing from dataset."
                )
            derived_col = target_col or "priceFactor"
            df[derived_col] = df[numerator_col] / df[baseline_col]
            df = df.replace({derived_col: {np.inf: np.nan, -np.inf: np.nan}})
            df = df[df[derived_col].notna()]
            df = df[df[derived_col] > 0]
            if numerator_col != derived_col and numerator_col in df.columns:
                df = df.drop(columns=[numerator_col])
            if baseline_col in df.columns:
                df = df.drop(columns=[baseline_col])
            target_col = derived_col
    else:
        if not target_col:
            target_col = base_target_col
        if target_col not in df.columns and base_target_col in df.columns:
            df = df.rename(columns={base_target_col: target_col})
        if baseline_col in df.columns:
            df = df.drop(columns=[baseline_col])

    for numeric_key in ["saleYear", "saleMonth", baseline_tx_col]:
        if numeric_key and numeric_key in df.columns:
            df[numeric_key] = pd.to_numeric(df[numeric_key], errors="coerce")

    preserve_columns = list(
        {
            target_col,
            "saleYear",
            "saleMonth",
            "suburb",
            *(resolved_config.get("force_keep", []) or []),
        }
    )
    if baseline_tx_col in df.columns and baseline_tx_col not in preserve_columns:
        preserve_columns.append(baseline_tx_col)

    df = _drop_low_variance(df, exclude=preserve_columns)

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != target_col]

    categorical_features = [
        col
        for col in df.columns
        if col not in numeric_features + [target_col]
    ]

    correlation_threshold = float(resolved_config.get("correlation_threshold", 0.9))
    correlated_numeric = _correlated_features(df, numeric_features, threshold=correlation_threshold)
    numeric_features = [col for col in numeric_features if col not in correlated_numeric]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),  # Using default behavior which varies by sklearn version
            ),
        ]
    )

    transformers: List[Tuple[str, Pipeline, List[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No features available for modelling after preprocessing.")

    preprocessor = ColumnTransformer(transformers)

    # Use a simple Decision Tree Regressor to get an idea of feature importance.
    # This is a fast way to get a sense of which features are most predictive.
    model = DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=6)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(df[numeric_features + categorical_features], df[target_col])

    # Aggregate feature importances from the model.
    # For one-hot encoded features, the importance of all derived binary features
    # are summed up to get the importance of the original categorical feature.
    preprocessor_step: ColumnTransformer = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor_step.get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    aggregated: Dict[str, float] = {}
    for name, importance in zip(feature_names, importances):
        if name.startswith("num__"):
            original = name.split("__", 1)[1]
        elif name.startswith("cat__"):
            remainder = name.split("__", 1)[1]
            original = remainder.split("_", 1)[0]
        else:
            original = name
        aggregated[original] = aggregated.get(original, 0.0) + float(importance)

    # Select features based on their importance score.
    # A threshold is determined based on the 15th most important feature,
    # and all features with an importance score above this threshold are selected.
    # A minimum of 5 features are selected, regardless of the threshold.
    sorted_features = sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    if sorted_features:
        threshold_index = min(len(sorted_features) - 1, 14)
        threshold = max(0.01, sorted_features[threshold_index][1])
        selected_features = [feat for feat, score in sorted_features if score >= threshold]
        minimum = min(5, len(sorted_features))
        if len(selected_features) < minimum:
            selected_features = [feat for feat, _ in sorted_features[:minimum]]
    else:
        selected_features = numeric_features + categorical_features

    # Ensure that a set of high-priority features are always included in the selected features.
    default_priority = [
        "street",
        "suburb",
        "propertyType",
        "bed",
        "bath",
        "car",
        "comparableCount",
        "saleYear",
        "saleMonth",
    ]
    priority_candidates = list(dict.fromkeys((resolved_config.get("force_keep", []) or []) + default_priority))
    priority_features = [col for col in priority_candidates if col in df.columns]
    for feature in priority_features:
        if feature not in selected_features:
            selected_features.append(feature)

    selected_features = sorted(set(selected_features))

    numeric_selected = [feat for feat in selected_features if feat in numeric_features]
    categorical_selected = [feat for feat in selected_features if feat in categorical_features]

    X = df[selected_features].copy()
    y = df[target_col].copy()

    # Store
    X_path = TRAINING_DIR / "X.parquet"
    y_path = TRAINING_DIR / "y.parquet"
    X.to_parquet(X_path, index=False)
    y.to_frame(name=target_col).to_parquet(y_path, index=False)

    feature_importance_path = TRAINING_DIR / "feature_importances.json"
    feature_importance_path.write_text(
        json.dumps(
            [
                {"feature": feat, "importance": score}
                for feat, score in sorted_features
            ],
            indent=2,
        )
    )

    categorical_levels = {
        col: sorted(value for value in X[col].unique())
        for col in categorical_selected
    }
    numeric_summary = {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "median": float(X[col].median()),
        }
        for col in numeric_selected
    }

    metadata = {
        "target": target_col,
        "raw_target": base_target_col,
        "target_type": "price_factor" if derive_factor else "regression",
        "baseline_lookup_keys": ["suburb", "saleYear", "saleMonth"],
        "baseline_transactions_column": baseline_tx_col
        if baseline_tx_col in df.columns
        else transactions_col_name,
        "selected_features": selected_features,
        "numeric_features": numeric_selected,
        "categorical_features": categorical_selected,
        "dropped_correlated_features": correlated_numeric,
        "config_excluded_columns": configured_exclusions,
        "manual_drop_columns": manual_existing,
        "rows": int(X.shape[0]),
        "categorical_levels": categorical_levels,
        "numeric_summary": numeric_summary,
        "baseline_history_file": HISTORY_FILENAME,
        "feature_config": resolved_config,
    }
    metadata_path = TRAINING_DIR / "feature_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return metadata


def recommend_features(df: pd.DataFrame, target_variable: str, exclude_columns: List[str] = None, score_threshold: float = 0.15) -> List[Dict[str, object]]:
    """
    Recommend features based on correlation with target variable and association strength.
    
    Args:
        df: The input DataFrame
        target_variable: The target variable to correlate with
        exclude_columns: List of columns to exclude from recommendations
        score_threshold: Minimum score threshold for including features
    
    Returns:
        List of recommended features with their scores and types
    """
    if exclude_columns is None:
        # Default exclude columns
        exclude_columns = ["streetAddress", "openInRpdata"]
    
    # Add target variable to exclude if it's provided
    if target_variable:
        exclude_columns = exclude_columns + [target_variable]
    
    # Get available columns by excluding specified columns
    available_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Separate column types
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col in available_columns]
    categorical_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns.tolist() if col in available_columns]
    
    recommended_features = []
    
    # Calculate correlation matrix for numeric columns
    if target_variable and target_variable in numeric_columns:
        numeric_df = df[numeric_columns].copy()
        corr_matrix_numeric = numeric_df.corr()
        
        # Add highly correlated numeric features
        target_corr = corr_matrix_numeric[target_variable].drop(target_variable, errors='ignore')
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
    
    # Add categorical features with strong associations to target
    if target_variable and categorical_columns:
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
        
        # Add categorical features with strong associations
        for result in anova_results:
            if result['Effect_Size'] > 0.01:  # Only include features with meaningful association
                recommended_features.append({
                    'Feature': result['Feature'],
                    'Score': result['Effect_Size'],
                    'Type': 'Categorical',
                    'Direction': 'association'
                })
    
    # Also include categorical-categorical associations if there are enough categorical variables
    if len(categorical_columns) >= 2:
        from scipy.stats import chi2_contingency
        
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
    
    # Sort recommended features by score
    recommended_features = sorted(recommended_features, key=lambda x: x['Score'], reverse=True)
    
    # Filter features based on the score threshold
    filtered_features = [f for f in recommended_features if f['Score'] > score_threshold]
    
    return filtered_features


if __name__ == "__main__":
    info = run_feature_selection()
    print("Feature selection completed. Selected features:")
    for feature in info["selected_features"]:
        print(f" - {feature}")