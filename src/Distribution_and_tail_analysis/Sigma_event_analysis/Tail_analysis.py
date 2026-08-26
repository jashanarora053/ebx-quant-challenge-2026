import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

DATA_DIR = r"C:\Users\omen\Downloads\archive"


def hill_estimator(returns, k=None):
    abs_ret = np.sort(np.abs(returns))[::-1]
   
    n = len(abs_ret)
    if k is None:
        k = int(np.sqrt(n))
    
    k = min(k, n - 1)
    
    if k < 2 or abs_ret[k] <= 0:
        return np.nan, k
    
    threshold = abs_ret[k]
    log_sum = np.sum(np.log(abs_ret[:k] / threshold))
    alpha = k / log_sum
    return round(alpha, 4), k

pooled_returns_1m = []
pooled_returns_5m = []
for day in range(1, 86):
    filepath = f"{DATA_DIR}/day{day}.csv"
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        price = df['Price']
        pooled_returns_1m.append(price.pct_change(60).dropna())
        pooled_returns_5m.append(price.pct_change(300).dropna())

# Compute Hill estimator for pooled 1-min and 5-min returns
hill_results = []

for horizon_name, pooled_list in [('1min', pooled_returns_1m), ('5min', pooled_returns_5m)]:
    pooled_ret = pd.concat(pooled_list, ignore_index=True).dropna().values
    
    alpha, k_used = hill_estimator(pooled_ret)
    
    hill_results.append({
        'scope': 'pooled',
        'horizon': horizon_name,
        'n_observations': len(pooled_ret),
        'k_tail_samples': k_used,
        'hill_tail_index': alpha,
        'excess_kurtosis': float(pd.Series(pooled_ret).kurtosis()),
    })

# Also compute per-day Hill for 1-min returns
for day in range(1, 86):
    filepath = f"{DATA_DIR}/day{day}.csv"
    if not os.path.exists(filepath):
        continue
    df = pd.read_csv(filepath)
    ret = df['Price'].pct_change(60).dropna().values
    if len(ret) < 100:
        continue
    alpha, k_used = hill_estimator(ret)
    hill_results.append({
        'scope': f'day_{day}',
        'horizon': '1min',
        'n_observations': len(ret),
        'k_tail_samples': k_used,
        'hill_tail_index': alpha,
        'excess_kurtosis': float(pd.Series(ret).kurtosis()),
    })

hill_df = pd.DataFrame(hill_results)
hill_df.to_csv("results/hill_tail_index.csv", index=False)