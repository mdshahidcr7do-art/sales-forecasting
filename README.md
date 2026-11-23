# Walmart Sales Forecasting

> **Status:** 🚧 In progress. Currently completing data cleaning and EDA.

Forecasting weekly sales across 45 Walmart stores using historical sales,
holiday events and economic indicators. The project compares statistical,
machine learning and deep learning approaches under a leakage-free,
time-based validation setup.

## Problem
Accurate demand forecasting helps retailers plan inventory and staffing.
Errors in holiday weeks are especially costly, so models are evaluated
with WMAE (holiday weeks weighted 5x) alongside MAE and RMSE.

## Approach
- Data loaded and merged with pandas (train + features + stores)
- EDA: trend, seasonality, holiday effects, decomposition, ACF/PACF
- Baselines: naive, seasonal naive, moving average
- Models: Ridge, Random Forest, LightGBM (lag, rolling and calendar features)
- Deep learning: LSTM
- Validation: time-based split (last 12 weeks held out), no random shuffling

## Results
Results will be added as experiments are completed.

## Key Findings
Coming soon.

## Tech Stack
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, statsmodels,
LightGBM, TensorFlow, Streamlit

## Project Structure
data/ · notebooks/ · src/ · app/ · requirements.txt

## How to Run
1. `pip install -r requirements.txt`
2. Download the dataset from Kaggle