# 02_analysis_and_charts.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / "SampleSuperstore_clean.csv", parse_dates=["order_date"], dayfirst=False)

# --- KPI summaries
kpis = {}
kpis["total_sales"] = df["sales"].sum()
kpis["total_profit"] = df["profit"].sum()
kpis["orders"] = df["order_id"].nunique() if "order_id" in df.columns else len(df)
kpis["avg_order_value"] = kpis["total_sales"] / kpis["orders"]
kpis["profit_margin"] = kpis["total_profit"] / kpis["total_sales"]
print(kpis)

# --- Sales trend (monthly)
df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
sales_month = df.groupby("month")["sales"].sum().reset_index()

plt.figure(figsize=(10,4))
sns.lineplot(data=sales_month, x="month", y="sales", marker="o")
plt.title("Monthly Sales Trend")
plt.tight_layout()
plt.savefig(OUT_DIR / "sales_trend.png")
plt.close()

# --- Top 10 products
top_products = df.groupby("product_name")["sales"].sum().nlargest(10).reset_index()
plt.figure(figsize=(8,5))
sns.barplot(x="sales", y="product_name", data=top_products, palette="viridis")
plt.title("Top 10 Products by Sales")
plt.tight_layout()
plt.savefig(OUT_DIR / "top_products.png")
plt.close()

# --- Sales by region (bar)
if "region" in df.columns:
    region_sales = df.groupby("region")["sales"].sum().reset_index()
    plt.figure(figsize=(7,4))
    sns.barplot(x="sales", y="region", data=region_sales, palette="magma")
    plt.title("Sales by Region")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sales_by_region.png")
    plt.close()

# --- Profit vs Sales scatter (identify low margin items)
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="sales", y="profit", alpha=0.4)
plt.title("Profit vs Sales (all orders)")
plt.tight_layout()
plt.savefig(OUT_DIR / "profit_vs_sales.png")
plt.close()

# --- Simple forecast (Holt-Winters)
ts = sales_month.set_index("month")["sales"]
try:
    model = ExponentialSmoothing(ts, trend="add", seasonal=None)
    fit = model.fit(optimized=True)
    forecast = fit.forecast(3)  # next 3 months
    forecast.to_csv(OUT_DIR / "sales_forecast.csv")
    # Plot forecast
    plt.figure(figsize=(10,4))
    plt.plot(ts.index, ts.values, label="Historical")
    plt.plot(forecast.index, forecast.values, label="Forecast", marker="o")
    plt.legend()
    plt.title("Sales Forecast (next 3 periods)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sales_forecast.png")
    plt.close()
except Exception as e:
    print("Forecasting error:", e)

print("Saved charts to", OUT_DIR)
