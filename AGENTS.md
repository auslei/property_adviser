# Propery Analysis

## Project
The python porject aimed at creating an accurate model, when provided with attributes (required, it will predict price in the coming year)

## Data
The data is located at ./data location in CSV format.

## Setup
Consider the project and create a venv with necessary libraries

## Process

1. Review the data and understand the structure
2. The data is on a suburb level
3. The street is import, so the preprocess should extract street from addresses
4. Prepare data and determine all attributes that are important for predicting prices
5. Pre-process data first:
    - column name cleansing (remove spaces, special characters, and name in camel case)
    - data cleansing:
        - remove columns that are mostly empty
        - reformat date into yyyymm 
        - impute values using median
    - store preprocessed data into ./data_preprocess
6. Feature Selection:
    - Using a method (logistic regression/decision tree) to find all features that are correlated with target (price next year)
    - Eliminate highly correlated features (picking based on the highest important)
    - ensure all features are convered into categorical features with numerical values, make sure bucket definition is reasonable for human to understand
    - store feature selection and target (X, y) into ./data_training
7. Model selection:
    - Go through all models and use training, validation dataset to find the best model to use.

8. Create a gui (streamlit) which allows select (categorical) attributes, to predict price