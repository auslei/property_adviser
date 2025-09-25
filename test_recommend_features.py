#!/usr/bin/env python3
"""
Script to test the recommend_features function from feature_selection module.
This loads a dataset and shows recommended features directly without GUI.
"""

import pandas as pd
from pathlib import Path
from src.feature_selection import recommend_features
from src.config import PREPROCESS_DIR


def main():
    # Load the derived data (or use cleaned.parquet from preprocessing)
    print("Loading dataset...")
    derived_path = PREPROCESS_DIR / "derived.parquet"
    
    if not derived_path.exists():
        # Try loading from cleaned data if derived doesn't exist
        derived_path = PREPROCESS_DIR / "cleaned.parquet"
        if not derived_path.exists():
            print(f"Dataset not found at {PREPROCESS_DIR}")
            print("Available files in PREPROCESS_DIR:")
            for file in PREPROCESS_DIR.glob("*.parquet"):
                print(f"  - {file.name}")
            return
    
    df = pd.read_parquet(derived_path)
    print(f"Dataset loaded with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Get numeric columns for target selection (excluding likely identifiers)
    exclude_cols = ["streetAddress", "openInRpdata", "parcelDetails"]
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Filter out likely identifier columns if they're numeric
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Numeric columns: {numeric_cols}")
    
    if not numeric_cols:
        print("No numeric columns found to use as target!")
        return
        
    # Use salePrice as target if available, otherwise use first numeric column
    if 'salePrice' in numeric_cols:
        target_variable = 'salePrice'
    else:
        target_variable = numeric_cols[0]  # Use first numeric column if salePrice not available
    print(f"Using '{target_variable}' as target variable")
    
    # Call the recommend_features function
    print(f"\nGetting recommendations for target: {target_variable}")
    recommended_features = recommend_features(
        df=df, 
        target_variable=target_variable, 
        exclude_columns=exclude_cols,
        score_threshold=0.0  # Get all features with any positive score
    )
    
    print(f"\nTotal recommended features: {len(recommended_features)}")
    print(f"Showing top 20 recommendations:")
    
    if recommended_features:
        # Convert to DataFrame for better display
        rec_df = pd.DataFrame(recommended_features)
        print(rec_df[['Feature', 'Score', 'Type']].head(20))
        
        # Show breakdown by type
        numeric_recs = [f for f in recommended_features if f['Type'] == 'Numeric']
        categorical_recs = [f for f in recommended_features if f['Type'] == 'Categorical']
        
        print(f"\nNumeric features recommended: {len(numeric_recs)}")
        print(f"Categorical features recommended: {len(categorical_recs)}")
        
        # Show high-scoring features (>0.15 which is the auto-select threshold)
        high_score_recs = [f for f in recommended_features if f['Score'] > 0.15]
        print(f"\nFeatures with score > 0.15 (auto-select threshold): {len(high_score_recs)}")
        if high_score_recs:
            high_df = pd.DataFrame(high_score_recs)
            print(high_df[['Feature', 'Score', 'Type']])
    else:
        print("No recommended features found!")


if __name__ == "__main__":
    main()