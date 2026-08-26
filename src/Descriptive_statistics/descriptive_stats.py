import pandas as pd    
import numpy as np    
import os            
import warnings
warnings.filterwarnings('ignore') 

DATA_DIR = r"C:\Users\omen\Downloads\archive"

def compute_descriptive_stats(series, label):
    clean = series.dropna()
    
    if len(clean) < 2:
        return {f'{label}_mean': np.nan, f'{label}_median': np.nan,
                f'{label}_std': np.nan, f'{label}_skew': np.nan,
                f'{label}_kurtosis': np.nan, f'{label}_count': 0}
        
    return {
        f'{label}_mean': clean.mean(),
        f'{label}_median': clean.median(),
        f'{label}_std':      clean.std(),
        f'{label}_skew':     clean.skew(), 
        f'{label}_kurtosis': clean.kurtosis(),     
        f'{label}_count':    len(clean),
    }


per_day_stats = []

pooled_prices = []
pooled_ret_1s = []
pooled_ret_1m = []
pooled_ret_5m = []

for day in range(1, 86):
    filepath = f"{DATA_DIR}/day{day}.csv"
    if not os.path.exists(filepath):
        continue
    
    df = pd.read_csv(filepath)
    price = df['Price']
    ret_1s = price.pct_change(1)
    ret_1m = price.pct_change(60)
    ret_5m = price.pct_change(300)
    
    row = {'day': day}
    row.update(compute_descriptive_stats(price,  'price'))
    row.update(compute_descriptive_stats(ret_1s, 'ret_1s'))
    row.update(compute_descriptive_stats(ret_1m, 'ret_1m'))
    row.update(compute_descriptive_stats(ret_5m, 'ret_5m'))
    per_day_stats.append(row)
    
    pooled_prices.append(price.dropna())
    pooled_ret_1s.append(ret_1s.dropna())
    pooled_ret_1m.append(ret_1m.dropna())
    pooled_ret_5m.append(ret_5m.dropna())
    
per_day_df = pd.DataFrame(per_day_stats)
pooled_prices_all = pd.concat(pooled_prices, ignore_index=True)
pooled_ret_1s_all = pd.concat(pooled_ret_1s, ignore_index=True)
pooled_ret_1m_all = pd.concat(pooled_ret_1m, ignore_index=True)
pooled_ret_5m_all = pd.concat(pooled_ret_5m, ignore_index=True)

pooled_summary = pd.DataFrame([
    compute_descriptive_stats(pooled_prices_all, 'price'),
    compute_descriptive_stats(pooled_ret_1s_all, 'ret_1s'),
    compute_descriptive_stats(pooled_ret_1m_all, 'ret_1m'),
    compute_descriptive_stats(pooled_ret_5m_all, 'ret_5m'),
])

per_day_df.to_csv("results/per_day_descriptive_stats.csv", index=False)
pooled_summary.to_csv("results/pooled_descriptive_stats.csv", index=False)
