import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

DATA_DIR = r"C:\Users\omen\Downloads\archive"

sigma_df = pd.read_csv("results/sigma_events_per_day.csv")
clustering_data = sigma_df[sigma_df['horizon'] == '1min'][['day', 'actual_3sigma', 'n_obs']].copy()
clustering_data['events_per_1000'] = (clustering_data['actual_3sigma'] / clustering_data['n_obs'] * 1000).round(2)


clustering_data = clustering_data.sort_values('actual_3sigma', ascending=False).reset_index(drop=True)


median_events = clustering_data['actual_3sigma'].median()
clustering_data['above_median'] = clustering_data['actual_3sigma'] > median_events


clustering_data_sorted = clustering_data.sort_values('day').reset_index(drop=True)
above = clustering_data_sorted['above_median']


runs = 1
for i in range(1, len(above)):
    if above.iloc[i] != above.iloc[i-1]:
        runs += 1

n_above = above.sum()
n_below = len(above) - n_above
n_total = len(above)
expected_runs = (2 * n_above * n_below) / n_total + 1
std_runs = np.sqrt((2 * n_above * n_below * (2 * n_above * n_below - n_total)) / (n_total**2 * (n_total - 1)))

runs_z = (runs - expected_runs) / std_runs if std_runs > 0 else 0

clustering_result = {
    'observed_runs': runs,
    'expected_runs': round(expected_runs, 2),
    'runs_z_score': round(runs_z, 4),
    'clustering_detected': runs_z < -1.96,  # 5% significance
    'median_3sigma_events': median_events,
}

clustering_data_sorted.to_csv("results/sigma_clustering_per_day.csv", index=False)
pd.DataFrame([clustering_result]).to_csv("results/sigma_clustering_test.csv", index=False)