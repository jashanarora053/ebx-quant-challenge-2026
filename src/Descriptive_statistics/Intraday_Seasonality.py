import pandas as pd    
import numpy as np    
import os            
import warnings
warnings.filterwarnings('ignore') 

DATA_DIR = r"C:\Users\omen\Downloads\archive"

per_day_bins = []

for day in range(1, 86):
    filepath = f"{DATA_DIR}/day{day}.csv"
    if not os.path.exists(filepath):
        continue

    df = pd.read_csv(filepath)
    df['abs_ret'] = df['Price'].pct_change(1).abs()

    df['seconds'] = df['Time'].apply(
        lambda t: int(t.split(':')[0])*3600 + int(t.split(':')[1])*60 + int(t.split(':')[2])
        if isinstance(t, str) and ':' in str(t) else np.nan
    )
    df = df.dropna(subset=['seconds', 'abs_ret'])
    df['bin_seconds'] = (df['seconds'] // 300) * 300

    day_grouped = df.groupby('bin_seconds').agg(
        day_mean_vol=('abs_ret', 'mean')
    ).reset_index()
    day_grouped['day'] = day
    per_day_bins.append(day_grouped)

all_bins = pd.concat(per_day_bins, ignore_index=True)

result = all_bins.groupby('bin_seconds').agg(
    day_count=('day', 'nunique'),
    mean_realized_1s_volatility=('day_mean_vol', 'mean'),
    std_across_days=('day_mean_vol', 'std')
).reset_index()

result.to_csv("volatility_seasonality.csv", index=False)
