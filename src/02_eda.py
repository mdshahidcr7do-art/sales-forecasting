from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "images"
IMG.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "cleaned.csv", parse_dates=["Date"])
sns.set_theme(style="whitegrid")

# 1. Trend: total weekly sales over time
total = df.groupby("Date")["Weekly_Sales"].sum()
plt.figure(figsize=(12, 4))
total.plot()
plt.title("Total weekly sales across all stores")
plt.ylabel("Sales ($)")
plt.tight_layout()
plt.savefig(IMG / "01_trend.png", dpi=120)
plt.close()

# 2. Seasonality: sales by week of year, one line per year
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
df["Year"] = df["Date"].dt.year
weekly = df.groupby(["Year", "Week"])["Weekly_Sales"].sum().reset_index()
plt.figure(figsize=(12, 4))
sns.lineplot(data=weekly, x="Week", y="Weekly_Sales", hue="Year")
plt.title("Sales by week of year (one line per year)")
plt.tight_layout()
plt.savefig(IMG / "02_seasonality.png", dpi=120)
plt.close()

# 3. Holiday effect
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="IsHoliday", y="Weekly_Sales", errorbar=None)
plt.title("Average weekly sales: holiday vs normal week")
plt.tight_layout()
plt.savefig(IMG / "03_holiday.png", dpi=120)
plt.close()

# 4. Store type
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="Type", y="Weekly_Sales", order=["A", "B", "C"], errorbar=None)
plt.title("Average weekly sales by store type")
plt.tight_layout()
plt.savefig(IMG / "04_store_type.png", dpi=120)
plt.close()

print("Charts saved to", IMG)