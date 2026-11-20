# Walmart Sales Forecasting

> **Status:** 🚧 In progress. Currently completing EDA and baseline models.

Forecasting weekly sales across 45 Walmart stores using historical sales,
holiday events and economic indicators. The project compares statistical,
machine learning and deep learning approaches under a leakage-free,
time-based validation setup.

## Problem
Accurate demand forecasting helps retailers plan inventory and staffing.
Errors in holiday weeks are especially costly, so models are evaluated
with WMAE (holiday weeks weighted 5x) alongside MAE and RMSE.

## Approach
- Data stored and queried in SQLite; tables joined with SQL
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
LightGBM, TensorFlow/PyTorch, SQLite, Streamlit

## Project Structure
data/ · notebooks/ · src/ · app/ · requirements.txt

## How to Run
1. `pip install -r requirements.txt`
2. Download the dataset from Kaggle into `data/`
3. Run the notebooks in order, or `streamlit run app/app.py`

## Limitations and Future Work
- Dataset covers 2010-2012 only; promotion (MarkDown) data is available only from late 2011
- Planned extensions: hierarchical forecasting (store/department level), LLM chat interface