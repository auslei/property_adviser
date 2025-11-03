
import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def find_data_files():
    """Find all csv and parquet files in the data directory."""
    return sorted(list(DATA_DIR.glob("**/*.csv")) + list(DATA_DIR.glob("**/*.parquet")))

@st.cache_data
def load_data(file_path):
    """Load data from a csv or parquet file."""
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    return None

def profile_dataframe(df):
    """Generate a profile of a dataframe."""
    st.write("### Data Preview")
    st.write(df.head())

    st.write("### Numerical Columns")
    numerical_cols = df.select_dtypes(include=["number"]).columns
    if len(numerical_cols) > 0:
        st.write(df[numerical_cols].describe().transpose())
    else:
        st.write("No numerical columns found.")

    st.write("### Categorical Columns")
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.write(f"#### {col}")
            st.write(df[col].value_counts())
    else:
        st.write("No categorical columns found.")

st.title("Data Profiler")

data_files = find_data_files()
if not data_files:
    st.warning("No data files found in the `data` directory.")
else:
    selected_file = st.selectbox("Select a data file", data_files)

    if st.button("Profile Data"):
        if selected_file:
            df = load_data(selected_file)
            if df is not None:
                profile_dataframe(df)
            else:
                st.error(f"Could not load data from {selected_file}")
