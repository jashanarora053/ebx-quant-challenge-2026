import pandas as pd   
import numpy as np     
import os
import warnings
warnings.filterwarnings('ignore') 

def parse_time_to_seconds(time_str):
    try:
        parts = str(time_str).split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])   
    except:
        return None

def check_sanity(df, day):    
    report = {'day': day}
    report['rows'] = df.shape[0]
    report['total_nans'] = int(df.isnull().sum().sum())
    report['price_nans'] = int(df['Price'].isnull().sum())
                               
    #TIME stamp Analysis
    seconds = df['Time'].apply(parse_time_to_seconds)

    malformed_mask = seconds.isna()    
    report['malformed_timestamps'] = int(malformed_mask.sum())
    valid_seconds = seconds.dropna().astype(int)
    
    report['duplicate_timestamps'] = int(valid_seconds.duplicated().sum())
    
    if len(valid_seconds) > 1:
        diffs = valid_seconds.diff().dropna()
        report['missing_seconds'] = int(diffs[diffs > 1].apply(lambda x: x - 1).sum())
        report['out_of_order'] = int((diffs <= 0).sum())
    else:
        report['missing_seconds'] = 0
        report['out_of_order'] = 0

    report['start_time'] = df['Time'].iloc[0]   
    report['end_time'] = df['Time'].iloc[-1]

    #PRICE Quality Analysis
    prices = df['Price'].dropna()
    report['zero_prices'] = int((prices == 0).sum())
    report['negative_prices'] = int((prices < 0).sum())
    report['infinite_prices'] = int(np.isinf(prices).sum())

    ## Flat lined streches check
    if len(prices) > 1:
        is_same = (prices.diff() == 0)
        groups = is_same.ne(is_same.shift()).cumsum()
        flat_groups = groups[is_same]
        if len(flat_groups) > 0:
            longest_flat = flat_groups.value_counts().max() + 1
            report['longest_flat_streak'] = int(longest_flat)
            report['total_zero_return_ticks'] = int(is_same.sum())
        else:
            report['longest_flat_streak'] = 1
            report['total_zero_return_ticks'] = 0

    ## Bad ticks check
    finite_prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(finite_prices) > 10:
        returns = finite_prices.pct_change().dropna()
        returns_std = returns.std()
        returns_mean = returns.mean()
        if returns_std > 0:
            z_scores = (returns - returns_mean).abs() / returns_std
            bad_ticks = int((z_scores > 5).sum())
            report['bad_ticks_5sigma'] = bad_ticks
            report['max_tick_pct_change'] = round(returns.abs().max() * 100, 4)
        else:
            report['bad_ticks_5sigma'] = 0
            report['max_tick_pct_change'] = 0.0
    
    return report


DATA_DIR = r"C:\Users\omen\Downloads\archive"
all_reports = []    
problem_days = []  

for day in range(1, 86):
    filepath = f"{DATA_DIR}/day{day}.csv"
    if not os.path.exists(filepath):
        problem_days.append({'day': day, 'issue': 'FILE NOT FOUND'})
        continue 
    
    df = pd.read_csv(filepath)       
    report = check_sanity(df, day)       
    all_reports.append(report)         

sanity_df = pd.DataFrame(all_reports)
sanity_df.to_csv("results/sanity_report.csv", index=False)
