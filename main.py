# main.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set page config first
st.set_page_config(page_title="Superstore Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

# --- Load Logo Path ---
logo_path = "data/maynard_logo.png"

# --- Sidebar Content ---
#st.sidebar.header("")

# Display the logo in the sidebar for a "circular" look
# We use a column structure in the sidebar to center the image,
# making it look contained and distinct from the filters.
col_logo, col_space = st.sidebar.columns([1, 2])
with col_logo:
    # Set the width to a specific size (e.g., 100) to keep it contained
    st.image(logo_path, width=100)

st.sidebar.header("Filters")
# The rest of the filters section follows here...


DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "Sample_Superstore.csv"


# --- Load + Clean Data ---
@st.cache_data
def load_and_clean_data(file_path):
    raw = pd.read_csv(file_path)
    df = raw.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    if {"sales", "quantity"} <= set(df.columns):
        df["unit_price"] = df["sales"] / df["quantity"]

    if {"profit", "sales"} <= set(df.columns):
        df["profit_margin"] = df["profit"] / df["sales"]

    for col in ["category", "sub_category", "region", "state", "city", "customer_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df.dropna(subset=["order_date"])  # remove rows with bad dates


df = load_and_clean_data(DATA_FILE)

# --- KPIs ---
total_sales = df["sales"].sum()
total_profit = df["profit"].sum()
orders = df["order_id"].nunique() if "order_id" in df.columns else len(df)
avg_order_value = total_sales / orders
profit_margin = total_profit / total_sales

st.title("📊 Superstore Analytics Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Orders", f"{orders:,}")
col4.metric("Avg Order Value", f"${avg_order_value:,.2f}")
col5.metric("Profit Margin", f"{profit_margin:.1%}")

# --- Filters (Using st.sidebar for the rest of the filters) ---
regions = df["region"].dropna().unique() if "region" in df.columns else []
selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=list(regions))
date_range = st.sidebar.date_input("Date Range", [df["order_date"].min(), df["order_date"].max()])

mask = df["order_date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
if selected_regions:
    mask &= df["region"].isin(selected_regions)

filtered_df = df.loc[mask]

# --- Sales Trend ---
st.subheader("📈 Monthly Sales Trend")
sales_month = filtered_df.groupby(pd.Grouper(key="order_date", freq="M"))["sales"].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=sales_month, x="order_date", y="sales", marker="o", ax=ax)
ax.set_title("Monthly Sales Trend")
st.pyplot(fig)

# --- Top Products ---
st.subheader("🏆 Top 10 Products")
top_products = filtered_df.groupby("product_name")["sales"].sum().nlargest(10).reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="sales", y="product_name", data=top_products, palette="viridis", ax=ax)
ax.set_title("Top 10 Products by Sales")
st.pyplot(fig)

# --- Sales by Region (PIE CHART) ---
if "region" in filtered_df.columns:
    st.subheader("🌎 Sales by Region Distribution")
    region_sales = filtered_df.groupby("region")["sales"].sum().reset_index()

    # Create the pie chart (circle graph)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Explode the largest slice for emphasis
    explode = [0.05 if sales == region_sales["sales"].max() else 0 for sales in region_sales["sales"]]

    ax.pie(
        region_sales["sales"],
        labels=region_sales["region"],
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set2"),
        explode=explode,
        shadow=True
    )

    ax.set_title("Sales Distribution by Region", fontsize=16)
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

# --- Profit vs Sales ---
st.subheader("💰 Profit vs Sales")
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=filtered_df, x="sales", y="profit", alpha=0.4, ax=ax)
ax.set_title("Profit vs Sales (orders)")
st.pyplot(fig)

# --- Forecast ---
st.subheader("🔮 Sales Forecast")
sales_month_ts = sales_month.set_index("order_date")["sales"]
try:
    model = ExponentialSmoothing(sales_month_ts, trend="add", seasonal=None)
    fit = model.fit(optimized=True)
    forecast = fit.forecast(3)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sales_month_ts.index, sales_month_ts.values, label="Historical")
    ax.plot(forecast.index, forecast.values, label="Forecast", marker="o")
    ax.set_title("Sales Forecast (next 3 months)")
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Forecasting error: {e}")

# --- Download filtered data ---
st.subheader("⬇️ Download Data")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Data as CSV", csv, "filtered_superstore.csv", "text/csv")