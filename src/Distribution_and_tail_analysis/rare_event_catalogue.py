import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"C:\Users\omen\Downloads\archive"
os.makedirs("results", exist_ok=True)

all_extreme_events = []

for day in range(1, 86):
    filepath = f"{DATA_DIR}/day{day}.csv"
    if not os.path.exists(filepath):
        continue
    
    df = pd.read_csv(filepath)
    
    df['ret_1m'] = df['Price'].pct_change(60)
    df['seconds'] = df['Time'].apply(
        lambda t: int(t.split(':')[0])*3600 + int(t.split(':')[1])*60 + int(t.split(':')[2])
        if isinstance(t, str) and ':' in str(t) else np.nan
    )
    df = df.dropna(subset=['seconds', 'ret_1m'])
    df['minute_bin'] = (df['seconds'] // 60).astype(int)
    
    tick_counts = df.groupby('minute_bin').size().rename('ticks_in_minute')
    df = df.merge(tick_counts, on='minute_bin', how='left')
    
    median_ticks = tick_counts.median()
    
    df['abs_ret_1m'] = df['ret_1m'].abs()
    df['day'] = day
    df['median_ticks_day'] = median_ticks
    df['tick_volume_ratio'] = (df['ticks_in_minute'] / median_ticks).round(2)

    worst_tick = df.loc[df['abs_ret_1m'].idxmax()]
    all_extreme_events.append(worst_tick)
    
    top3 = df.nlargest(3, 'abs_ret_1m')
    for _, row in top3.iterrows():
        all_extreme_events.append(row)

catalogue_df = pd.DataFrame(all_extreme_events)

catalogue_df = catalogue_df.drop_duplicates(subset=['day', 'Time']).reset_index(drop=True)

catalogue_df = catalogue_df.nlargest(20, 'abs_ret_1m').reset_index(drop=True)


final_catalogue = catalogue_df[['day', 'Time', 'Price', 'ret_1m', 'abs_ret_1m', 
                                  'ticks_in_minute', 'median_ticks_day', 'tick_volume_ratio']].copy()

final_catalogue['ret_1m_pct'] = (final_catalogue['ret_1m'] * 100).round(4)
final_catalogue['direction'] = final_catalogue['ret_1m'].apply(lambda x: 'UP' if x > 0 else 'DOWN')
final_catalogue['volume_spike'] = final_catalogue['tick_volume_ratio'] > 1.5

final_catalogue = final_catalogue.sort_values('abs_ret_1m', ascending=False).reset_index(drop=True)
final_catalogue.index = final_catalogue.index + 1  # Rank starting from 1
final_catalogue.index.name = 'rank'

final_catalogue.to_csv("results/rare_event_catalogue.csv")
