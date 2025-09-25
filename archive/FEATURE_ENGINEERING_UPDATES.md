# Feature Engineering Page - Current State

## Overview
The Feature Engineering page (`app/pages/2_Feature_Engineering.py`) provides a comprehensive and user-friendly approach to feature engineering for property price prediction.

## Architectural Principles
1. **Module Responsibility**: All complex calculations should be done in corresponding modules, not in the Streamlit app. Calculations should only be performed in the Streamlit app if they are specifically for charting and simple.
2. **Data Derivation**: All data derivation and preprocessing should be done in preprocessing steps, not in other modules. This ensures derived features like medians and price factors are precomputed and available in the derived dataset.

## Current Capabilities:

### 1. **Comprehensive Feature Analysis**
- Numeric feature correlation analysis using Pearson correlation coefficient
- Categorical feature association analysis using ANOVA (effect size: eta squared)
- Categorical-categorical association analysis using Cramér's V with threshold filtering
- Visualization for both numeric and categorical feature importance

### 2. **Intelligent Feature Recommendation**
- Automatically calculates and displays recommended features based on correlation/association strength
- Shows top features with scores for both numeric and categorical variables
- Provides default selection of the most relevant features
- "Auto Select Features" button for automated feature selection based on recommendations

### 3. **Enhanced User Interface**
- Tabbed interface for numeric and categorical feature analysis
- Clean, intuitive interface with proper section organization
- Expandable sections for detailed information
- Clear feedback messages for user actions

### 4. **Training Dataset Generation**
- "Generate Training Datasets (X, y)" button that runs the feature selection process
- Creates the necessary training files (X.parquet, y.parquet) with the selected features
- Provides preview of generated datasets and selected features
- Clear success/error messaging for dataset generation

## Technical Implementation:

1. **Statistical Analysis Methods**:
   - Numeric-Numeric: Pearson correlation coefficient
   - Categorical-Numeric: ANOVA with effect size (eta squared) measurement
   - Categorical-Categorical: Cramér's V with threshold filtering (> 0.1 for meaningful associations)

2. **Feature Selection Process**:
   - Recommends features based on statistical significance and correlation/association strength
   - Single multiselect for all feature selection with recommended defaults
   - Generates training datasets using the underlying `src.feature_selection.run_feature_selection()` function
   - Properly configures feature selection with user's chosen features

3. **Dataset Generation**:
   - Creates X.parquet, y.parquet, and metadata files in the training directory
   - Ensures baseline key columns (saleYear, saleMonth, suburb) are preserved for model training
   - Provides preview of datasets shape and sample data

4. **Configuration Handling**:
   - Uses consolidated `config/features.yml` for feature engineering configuration
   - Properly loads and saves configuration settings
   - Maintains backward compatibility with existing pipeline

## How to Use:

1. **Analyze Features**: 
   - Review the "Numeric Features" tab for correlation analysis
   - Review the "Categorical Features" tab for association analysis
   - Check the "Categorical-Categorical Associations" expandable section for variable relationships

2. **Review Recommendations**: 
   - View automatically calculated recommended features based on statistical strength
   - Use the "Auto Select Features" button for intelligent feature selection

3. **Select Features**: 
   - Use the multiselect to choose features for model training
   - Manually add/remove features as needed

4. **Generate Datasets**: 
   - Click "Generate Training Datasets (X, y)" to create training files
   - Review the preview of generated datasets

5. **Proceed**: 
   - Move to the "3 Model Selection" page for model training and comparison

The Feature Engineering page provides a complete solution for understanding, selecting, and preparing features for property price prediction models.