from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# 1. Load
train = pd.read_csv(DATA / "train.csv")
features = pd.read_csv(DATA / "features.csv")
stores = pd.read_csv(DATA / "stores.csv")
print("Shapes:", train.shape, features.shape, stores.shape)

# 2. Merge
df = train.merge(stores, on="Store", how="left")
df = df.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
print("Merged shape (should be 421570 rows):", df.shape)

# 3. Quality checks
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicates:", df.duplicated(subset=["Store", "Dept", "Date"]).sum())
print("Negative sales:", (df["Weekly_Sales"] < 0).sum())
print("Date range:", df["Date"].min(), "to", df["Date"].max())

# 4. Cleaning
markdowns = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
df[markdowns] = df[markdowns].fillna(0)

df["Is_Negative"] = df["Weekly_Sales"] < 0

# 5. Series length per (Store, Dept)
series = df.groupby(["Store", "Dept"]).size()
print("\nWeeks per series:")
print(series.describe())
print("Series with fewer than 52 weeks:", (series < 52).sum())

# 6. Remaining missing values after cleaning
remaining = df.isnull().sum()
print("\nRemaining missing values:")
print(remaining[remaining > 0])

# 7. Save the cleaned data (last step)
df.to_csv(DATA / "cleaned.csv", index=False)
print("\nSaved", DATA / "cleaned.csv")