import pandas as pd
import numpy as np
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"C:\Users\omen\Downloads\archive"

sample_days = [1, 20, 40, 60, 85]

results = []

for day in sample_days:
    filepath = f"{DATA_DIR}/day{day}.csv"
    if not os.path.exists(filepath):
        continue
    
    df = pd.read_csv(filepath)
    price = df['Price']
    
    for horizon_name, horizon_periods in [('1min', 60), ('5min', 300)]:
        ret = price.pct_change(horizon_periods).dropna()
        
        if len(ret) < 20:
            continue
        
        row = {
            'scope': f'day_{day}',
            'return_horizon': horizon_name,
            'n_observations': len(ret),
            'mean': ret.mean(),
            'std': ret.std(),
            'skewness': ret.skew(),
            'kurtosis': ret.kurtosis(), 
        }
        
        # Test 1: Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(ret)
        row['jarque_bera_stat'] = jb_stat
        row['jarque_bera_pvalue'] = jb_p
        
        # Test 2: D'Agostino K-squared
        dag_stat, dag_p = stats.normaltest(ret)
        row['dagostino_k2_stat'] = dag_stat
        row['dagostino_k2_pvalue'] = dag_p
        
        # Test 3: Anderson-Darling
        ad_result = stats.anderson(ret, dist='norm')
        row['anderson_darling_stat'] = ad_result.statistic
        row['anderson_darling_cv_5pct'] = ad_result.critical_values[2]  # 5% level
        row['anderson_darling_reject_5pct'] = ad_result.statistic > ad_result.critical_values[2]
        
        # Test 4: Shapiro-Wilk (limited to 5000 samples)
        sample = ret.sample(min(5000, len(ret)), random_state=42)
        sw_stat, sw_p = stats.shapiro(sample)
        row['shapiro_wilk_stat'] = sw_stat
        row['shapiro_wilk_pvalue'] = sw_p
        
        results.append(row)

# ---- Pooled tests (all 85 days combined) ----
all_ret_1m = []
all_ret_5m = []

for day in range(1, 86):
    filepath = f"{DATA_DIR}/day{day}.csv"
    if not os.path.exists(filepath):
        continue
    df = pd.read_csv(filepath)
    price = df['Price']
    all_ret_1m.append(price.pct_change(60).dropna())
    all_ret_5m.append(price.pct_change(300).dropna())

pooled_1m = pd.concat(all_ret_1m, ignore_index=True)
pooled_5m = pd.concat(all_ret_5m, ignore_index=True)

for horizon_name, ret in [('1min', pooled_1m), ('5min', pooled_5m)]:
    row = {
        'scope': 'pooled_85_days',
        'return_horizon': horizon_name,
        'n_observations': len(ret),
        'mean': ret.mean(),
        'std': ret.std(),
        'skewness': ret.skew(),
        'kurtosis': ret.kurtosis(),
    }
    
    jb_stat, jb_p = stats.jarque_bera(ret)
    row['jarque_bera_stat'] = jb_stat
    row['jarque_bera_pvalue'] = jb_p
    
    dag_stat, dag_p = stats.normaltest(ret)
    row['dagostino_k2_stat'] = dag_stat
    row['dagostino_k2_pvalue'] = dag_p
    
    ad_result = stats.anderson(ret, dist='norm')
    row['anderson_darling_stat'] = ad_result.statistic
    row['anderson_darling_cv_5pct'] = ad_result.critical_values[2]
    row['anderson_darling_reject_5pct'] = ad_result.statistic > ad_result.critical_values[2]
    
    sample = ret.sample(min(5000, len(ret)), random_state=42)
    sw_stat, sw_p = stats.shapiro(sample)
    row['shapiro_wilk_stat'] = sw_stat
    row['shapiro_wilk_pvalue'] = sw_p
    
    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv("normality_tests.csv", index=False)
