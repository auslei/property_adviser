import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.common.app_utils import load_model_resources
from src.model_prediction import predict_property_price, predict_with_confidence_interval, predict_property_prices_batch


st.set_page_config(page_title="Model Prediction", layout="wide")
st.title("Property Price Prediction")
st.caption(
    "Use the trained model to predict property prices based on input features."
)


# Load model and metadata
try:
    feature_metadata, model, model_summary = load_model_resources()
    st.success("Model loaded successfully!")
    
    # Show model information
    st.write(f"**Model Type:** {model_summary.get('best_model', 'Unknown')}")
    st.write(f"**Target Variable:** {feature_metadata.get('target', 'Unknown')}")
    st.write(f"**Features Used:** {len(feature_metadata.get('selected_features', []))} total")
    
except FileNotFoundError as e:
    st.error(f"Model not found: {e}")
    st.info("Please run the full pipeline (preprocessing → feature selection → model training) first.")
    st.stop()


# Single prediction interface
st.subheader("Single Property Prediction")

# Create input fields for the required features
col1, col2, col3 = st.columns(3)
with col1:
    # Year and Month Input
    sale_year = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
    sale_month = st.selectbox("Month", options=list(range(1, 13)), index=5)  # Default to June
    yearmonth = sale_year * 100 + sale_month

with col2:
    bed = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
    bath = st.number_input("Bathrooms", min_value=0, max_value=20, value=2)

with col3:
    car = st.number_input("Car Spaces", min_value=0, max_value=10, value=2)
    property_types = ["House", "Unit", "Apartment", "Townhouse"]  # Common types
    default_type = "House"
    property_type = st.selectbox("Property Type", options=property_types, index=property_types.index(default_type) if default_type in property_types else 0)
    
street = st.text_input("Street Name", value="Example Street")

# Prediction button
if st.button("Predict Price", type="primary"):
    with st.spinner("Making prediction..."):
        try:
            # Make prediction
            predicted_price = predict_property_price(
                yearmonth=yearmonth,
                bed=bed,
                bath=bath,
                car=car,
                property_type=property_type,
                street=street
            )
            
            # Also get prediction with confidence interval
            conf_result = predict_with_confidence_interval(
                yearmonth=yearmonth,
                bed=bed,
                bath=bath,
                car=car,
                property_type=property_type,
                street=street
            )
            
            # Display results
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Price", f"${predicted_price:,.2f}")
            col2.metric("Lower Bound (95%)", f"${conf_result['lower_bound']:,.2f}")
            col3.metric("Upper Bound (95%)", f"${conf_result['upper_bound']:,.2f}")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")


# Batch prediction interface
st.subheader("Batch Property Predictions")
st.write("Upload a CSV file with multiple properties to predict prices for.")

# File uploader for batch prediction
uploaded_file = st.file_uploader(
    "Upload CSV file", 
    type=["csv"],
    help="CSV should contain columns: saleYear, saleMonth, bed, bath, car, propertyType, street"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Show uploaded data
        st.write("Uploaded Data:")
        st.dataframe(df.head())
        
        # Check required columns
        required_cols = ['saleYear', 'saleMonth', 'bed', 'bath', 'car', 'propertyType', 'street']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            # Convert to yearmonth format
            df['yearmonth'] = df['saleYear'] * 100 + df['saleMonth']
            
            # Prepare data for prediction
            properties = df[required_cols].to_dict('records')
            
            if st.button("Predict All Prices"):
                with st.spinner("Making batch predictions..."):
                    try:
                        prices = predict_property_prices_batch(properties)
                        
                        # Add predictions to dataframe
                        results_df = df.copy()
                        results_df['predicted_price'] = prices
                        results_df['predicted_price_formatted'] = results_df['predicted_price'].apply(lambda x: f"${x:,.2f}")
                        
                        st.success(f"Completed {len(prices)} predictions!")
                        
                        # Show results
                        st.write("Prediction Results:")
                        display_cols = required_cols + ['predicted_price_formatted']
                        st.dataframe(results_df[display_cols])
                        
                        # Download button for results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="property_price_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
                        
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")


# Model information section
st.subheader("Model Information")
with st.expander("View Model Details"):
    st.json(model_summary)
    
    st.write("**Selected Features:**")
    selected_features = feature_metadata.get('selected_features', [])
    for i, feature in enumerate(selected_features[:20]):  # Show first 20 features
        st.write(f"- {feature}")
    if len(selected_features) > 20:
        st.write(f"... and {len(selected_features) - 20} more")