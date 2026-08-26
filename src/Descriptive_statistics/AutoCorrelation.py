import pandas as pd    
import numpy as np    
import os            
import warnings
warnings.filterwarnings('ignore') 

DATA_DIR = r"C:\Users\omen\Downloads\archive"

lags = range(1, 61)
acf_per_day = []

for day in range(1, 86):
    filepath = f"{DATA_DIR}/day{day}.csv"
    if not os.path.exists(filepath):
        continue
    
    df = pd.read_csv(filepath)
    ret = df['Price'].pct_change(1).dropna()
    day_acf = {}
    for lag in lags:
        day_acf[f'lag_{lag}'] = ret.autocorr(lag=lag)
    acf_per_day.append(day_acf)


acf_df = pd.DataFrame(acf_per_day)
avg_acf = acf_df.mean()

acf_results = pd.DataFrame({
    'lag_seconds': list(range(1, 61)),
    'avg_autocorrelation': [avg_acf[f'lag_{lag}'] for lag in range(1, 61)]
})
acf_results.to_csv("acf_returns_lag1_to_60.csv", index=False)
