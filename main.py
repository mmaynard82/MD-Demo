# main.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import sys

# Set page config first
st.set_page_config(page_title="Superstore Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

# --- FILE PATH SETUP ---
# Use pathlib for clean, OS-agnostic path management
# Get the absolute directory where this script (main.py) is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# CRITICAL FIX: Ensure the case matches EXACTLY what is on GitHub
LOGO_PATH = SCRIPT_DIR / "data" / "logo.png"
DATA_FILE = SCRIPT_DIR / "data" / "Sample_Superstore.csv"

# Optional: Add project root to sys path for local imports
if SCRIPT_DIR not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

# --- Logo Display ---
# Display the logo on the main page (Optional: may look better just in sidebar)
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=100)
else:
    # This warning helps confirm the path being used on Streamlit Cloud
    st.warning(f"Logo not found at {LOGO_PATH}. Please check filename case (e.g., 'Maynard_logo.png').")

# --- Sidebar Content ---
st.sidebar.header("Filters")

# Display the logo in the sidebar for a "circular" look
col_logo, col_space = st.sidebar.columns([1, 2])
with col_logo:
    if LOGO_PATH.exists():
        # Pass the path as a string when using st.image
        st.image(str(LOGO_PATH), width=100)


# --- Load + Clean Data ---
@st.cache_data
def load_and_clean_data(file_path):
    # CRITICAL: Ensure the file exists before attempting to read
    if not file_path.exists():
        st.error(f"Data file not found at: {file_path}")
        return pd.DataFrame() # Return empty DataFrame on failure

    raw = pd.read_csv(file_path)
    df = raw.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    if {"sales", "quantity"} <= set(df.columns):
        # Handle potential division by zero if quantity is 0
        df["unit_price"] = df.apply(lambda row: row['sales'] / row['quantity'] if row['quantity'] != 0 else 0, axis=1)

    if {"profit", "sales"} <= set(df.columns):
        # Handle potential division by zero if sales is 0
        df["profit_margin"] = df.apply(lambda row: row['profit'] / row['sales'] if row['sales'] != 0 else 0, axis=1)

    for col in ["category", "sub_category", "region", "state", "city", "customer_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df.dropna(subset=["order_date"])  # remove rows with bad dates


df = load_and_clean_data(DATA_FILE)

# Guard clause to prevent errors if data loading fails
if df.empty:
    st.stop()


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
# Ensure date and region columns exist before filtering
if "region" not in df.columns or "order_date" not in df.columns:
    st.error("Missing critical columns (region or order_date) after cleaning.")
    st.stop()

regions = df["region"].dropna().unique()
selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=list(regions))
date_range = st.sidebar.date_input("Date Range", [df["order_date"].min().date(), df["order_date"].max().date()])

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
if "product_name" in filtered_df.columns:
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

    fig, ax = plt.subplots(figsize=(8, 8))
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
    ax.axis("equal")
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
    # Ensure there are enough data points for the model
    if len(sales_month_ts) >= 3:
        model = ExponentialSmoothing(sales_month_ts, trend="add", seasonal=None)
        fit = model.fit(optimized=True)
        forecast = fit.forecast(3)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sales_month_ts.index, sales_month_ts.values, label="Historical")
        ax.plot(forecast.index, forecast.values, label="Forecast", marker="o")
        ax.set_title("Sales Forecast (next 3 months)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Not enough data to generate a meaningful sales forecast.")
except Exception as e:
    st.warning(f"Forecasting error: {e}")

# --- Download filtered data ---
st.subheader("⬇️ Download Data")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Data as CSV", csv, "filtered_superstore.csv", "text/csv")