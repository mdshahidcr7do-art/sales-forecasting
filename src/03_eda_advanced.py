from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "images"
IMG.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "cleaned.csv", parse_dates=["Date"])
sns.set_theme(style="whitegrid")

# 5. Top departments by total sales
top = df.groupby("Dept")["Weekly_Sales"].sum().nlargest(10)
plt.figure(figsize=(8, 4))
sns.barplot(x=top.index.astype(str), y=top.values)
plt.title("Top 10 departments by total sales")
plt.xlabel("Dept")
plt.tight_layout()
plt.savefig(IMG / "05_top_depts.png", dpi=120)
plt.close()

# 6. Decomposition of total weekly sales
total = df.groupby("Date")["Weekly_Sales"].sum()
result = seasonal_decompose(total, model="additive", period=52)
fig = result.plot()
fig.set_size_inches(12, 8)  
plt.tight_layout()
plt.savefig(IMG / "06_decomposition.png", dpi=120)
plt.close()

# 7. ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(total, lags=52, ax=ax[0])
plot_pacf(total, lags=52, ax=ax[1])
plt.tight_layout()
plt.savefig(IMG / "07_acf_pacf.png", dpi=120)
plt.close()

# 8. External features vs sales
cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
plt.figure(figsize=(6, 5))
sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation: sales vs external features")
plt.tight_layout()
plt.savefig(IMG / "08_correlation.png", dpi=120)
plt.close()

print("Done")