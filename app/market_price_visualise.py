import altair as alt
import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(layout="wide")

# Title
st.title("Market Price Visualisation")

# Load data
@st.cache_data
def load_data():
    df = pd.read_parquet("data/derive/derived2.parquet")
    return df

df = load_data()

# Filters
st.sidebar.header("Filters")

year_range = st.sidebar.slider("Select Year Range",
                               min_value=int(df["saleYear"].min()),
                               max_value=int(df["saleYear"].max()),
                               value=(int(df["saleYear"].min()), int(df["saleYear"].max())))

st.title(f"Year Range: {year_range}")

T = st.sidebar.selectbox("", options=["All", "Unit", "House"], index=0)

def create_filter(column_name, default_value="All"):
    options = ["All"] + df[column_name].unique().tolist()
    index = 0 if default_value == "All" else options.index(default_value)
    return st.sidebar.selectbox(f"Select {column_name}", options, index=index)

bed_bucket = create_filter("bed_bucket")
bath_bucket = create_filter("bath_bucket")
land_bucket = create_filter("land_bucket")
floor_bucket = create_filter("floor_bucket")
street = create_filter("street")
propertyType = create_filter("propertyType")

# Filter data
filtered_df = df.copy()

if bed_bucket != "All":
    filtered_df = filtered_df[filtered_df["bed_bucket"] == bed_bucket]
if bath_bucket != "All":
    filtered_df = filtered_df[filtered_df["bath_bucket"] == bath_bucket]
if land_bucket != "All":
    filtered_df = filtered_df[filtered_df["land_bucket"] == land_bucket]
if floor_bucket != "All":
    filtered_df = filtered_df[filtered_df["floor_bucket"] == floor_bucket]
if street != "All":
    filtered_df = filtered_df[filtered_df["street"] == street]
if propertyType != "All":
    filtered_df = filtered_df[filtered_df["propertyType"] == propertyType]

filtered_df = filtered_df[
    (filtered_df["saleYear"] >= year_range[0]) &
    (filtered_df["saleYear"] <= year_range[1])
]


def year_month_to_year_quarter(ym):
    year = ym // 100
    month = ym % 100
    quarter = (month - 1) // 3 + 1
    return f"{year}-Q{quarter}"

filtered_df['saleYearQuarter'] = filtered_df['saleYearMonth'].apply(year_month_to_year_quarter)

print(filtered_df.head())

# Aggregate data
agg_df = filtered_df.groupby('saleYearQuarter').agg(
    salePrice_mean=('salePrice', 'mean'),
    salePrice_median=('salePrice', 'median')
).reset_index()

# Chart
st.header("Price Trends")

# Reshape data for grouped bar chart
chart_data = agg_df.melt(
    id_vars=['saleYearQuarter'],
    value_vars=['salePrice_mean', 'salePrice_median'],
    var_name='metric',
    value_name='price'
)

print(chart_data.head())

# Rename metrics for display
chart_data['metric'] = chart_data['metric'].map({
    'salePrice_mean': 'Average Sale Price',
    'salePrice_median': 'Median Sale Price'
})

chart = alt.Chart(chart_data).mark_bar().encode(
    x=alt.X('saleYearQuarter:O', title='Sale Year-Quarter'),
    y=alt.Y('price:Q', title='Sale Price', axis=alt.Axis(format='$,.0f')),
    color=alt.Color('metric:N',
                    scale=alt.Scale(domain=['Average Sale Price', 'Median Sale Price'],
                                  range=['blue', 'red']),
                    legend=alt.Legend(title='Metric')),
    xOffset='metric:N'
)

st.altair_chart(chart, use_container_width=True)

# table showing the top 20 streets by mean saleprice
st.header("Top 20 Streets by Mean Sale Price")
top_streets = (filtered_df.groupby('street')
               .agg(mean_sale_price=('salePrice', 'mean'), meadian_sale_price=('salePrice', 'median'))
               .reset_index()
               .sort_values(by='mean_sale_price', ascending=False)
               .head(20))
st.table(top_streets.style.format({'mean_sale_price': '${:,.2f}', 'meadian_sale_price': '${:,.2f}'}))