import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

DATA_DIR = r"C:\Users\omen\Downloads\archive"

sigma_levels = [1, 2, 3, 4, 5]

theoretical_pct = {
    1: 1 - (stats.norm.cdf(1) - stats.norm.cdf(-1)),   
    2: 1 - (stats.norm.cdf(2) - stats.norm.cdf(-2)),  
    3: 1 - (stats.norm.cdf(3) - stats.norm.cdf(-3)),   
    4: 1 - (stats.norm.cdf(4) - stats.norm.cdf(-4)),  
    5: 1 - (stats.norm.cdf(5) - stats.norm.cdf(-5)),  
}

per_day_sigma = []
pooled_returns_1m = []
pooled_returns_5m = []

for day in range(1, 86):
    filepath = f"{DATA_DIR}/day{day}.csv"
    if not os.path.exists(filepath):
        continue
    
    df = pd.read_csv(filepath)
    price = df['Price']
    
    for horizon_name, periods in [('1min', 60), ('5min', 300)]:
        ret = price.pct_change(periods).dropna()
        
        if len(ret) < 20:
            continue
        
        mu = ret.mean()
        sigma = ret.std()
        
        z_scores = (ret - mu).abs() / sigma
    
        
        n = len(ret)
        row = {'day': day, 'horizon': horizon_name, 'n_obs': n}
        
        for k in sigma_levels:
            actual_count = int((z_scores > k).sum())
           
            expected_count = theoretical_pct[k] * n

            
            ratio = actual_count / expected_count if expected_count > 0 else np.nan
            
            
            row[f'actual_{k}sigma'] = actual_count
            row[f'expected_{k}sigma'] = round(expected_count, 2)
            row[f'ratio_{k}sigma'] = round(ratio, 2)
        
        per_day_sigma.append(row)
        
        if horizon_name == '1min':
            pooled_returns_1m.append(ret)
        else:
            pooled_returns_5m.append(ret)

sigma_df = pd.DataFrame(per_day_sigma)

pooled_sigma_rows = []
for horizon_name, pooled_list in [('1min', pooled_returns_1m), ('5min', pooled_returns_5m)]:
    pooled_ret = pd.concat(pooled_list, ignore_index=True)
    mu = pooled_ret.mean()
    sigma = pooled_ret.std()
    z_scores = (pooled_ret - mu).abs() / sigma
    n = len(pooled_ret)
    
    row = {'day': 'pooled', 'horizon': horizon_name, 'n_obs': n}
    for k in sigma_levels:
        actual = int((z_scores > k).sum())
        expected = theoretical_pct[k] * n
        row[f'actual_{k}sigma'] = actual
        row[f'expected_{k}sigma'] = round(expected, 2)
        row[f'ratio_{k}sigma'] = round(actual / expected if expected > 0 else np.nan, 2)
    pooled_sigma_rows.append(row)

pooled_sigma_df = pd.DataFrame(pooled_sigma_rows)

sigma_df.to_csv("sigma_events_per_day.csv", index=False)
pooled_sigma_df.to_csv("sigma_events_pooled.csv", index=False)
