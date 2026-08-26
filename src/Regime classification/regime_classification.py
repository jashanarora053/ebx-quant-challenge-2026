import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"C:\Users\omen\Downloads\archive"
os.makedirs("results", exist_ok=True)

def hurst_rs(prices):
    prices = np.array(prices)
    n = len(prices)
    
    if n < 100:
        return np.nan

    returns = np.diff(np.log(prices))
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 100:
        return np.nan
    
    chunk_sizes = []
    rs_values = []
    
    for chunk_size in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if chunk_size >= len(returns):
            break
        
        n_chunks = len(returns) // chunk_size
        if n_chunks < 1:
            break
        
        rs_list = []
        for i in range(n_chunks):
            chunk = returns[i * chunk_size : (i + 1) * chunk_size]
            mean_chunk = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_chunk)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk, ddof=1)
            
            if S > 0:
                rs_list.append(R / S)
        
        if len(rs_list) > 0:
            chunk_sizes.append(chunk_size)
            rs_values.append(np.mean(rs_list))
    
    if len(chunk_sizes) < 2:
        return np.nan
   
    log_sizes = np.log(chunk_sizes)
    log_rs = np.log(rs_values)
    
    H = np.polyfit(log_sizes, log_rs, 1)[0]
    
    return round(H, 4)


def variance_ratio(prices, q=5):
   
    prices = np.array(prices)
    log_prices = np.log(prices[prices > 0])
    
    n = len(log_prices)
    if n < q + 10:
        return np.nan, np.nan
    
    ret_1 = np.diff(log_prices)
    ret_q = log_prices[q:] - log_prices[:-q]

    var_1 = np.var(ret_1, ddof=1)
    var_q = np.var(ret_q, ddof=1)
    
    if var_1 == 0:
        return np.nan, np.nan
    
    vr = var_q / (q * var_1)
    
    nq = len(ret_1)
    z_stat = (vr - 1) / np.sqrt(2 * (q - 1) / nq)
    
    return round(vr, 4), round(z_stat, 4)


HURST_MR_THRESHOLD = 0.45      # Below this = Mean-Reverting
HURST_MOM_THRESHOLD = 0.55     # Above this = Momentum
VR_MR_THRESHOLD = 0.95         # Below this = Mean-Reverting
VR_MOM_THRESHOLD = 1.05        # Above this = Momentum

def classify_regime(hurst, vr):
    hurst_label = 'Unknown'
    vr_label = 'Unknown'
    
    if np.isnan(hurst):
        hurst_label = 'Unknown'
    elif hurst < HURST_MR_THRESHOLD:
        hurst_label = 'Mean-Reverting'
    elif hurst > HURST_MOM_THRESHOLD:
        hurst_label = 'Momentum'
    else:
        hurst_label = 'Random Walk'
    
    if np.isnan(vr):
        vr_label = 'Unknown'
    elif vr < VR_MR_THRESHOLD:
        vr_label = 'Mean-Reverting'
    elif vr > VR_MOM_THRESHOLD:
        vr_label = 'Momentum'
    else:
        vr_label = 'Random Walk'
    
    if hurst_label == vr_label:
        final = hurst_label
        agreement = 'Agree'
    else:
        final = hurst_label
        agreement = 'Disagree'
    
    return hurst_label, vr_label, final, agreement

results = []

for day in range(1, 86):
    filepath = f"{DATA_DIR}/day{day}.csv"
    if not os.path.exists(filepath):
        continue
    
    df = pd.read_csv(filepath)
    prices = df['Price'].dropna().values
    
    if len(prices) < 200:
        continue
    
    hurst = hurst_rs(prices)
    vr, vr_z = variance_ratio(prices, q=5)
    
    hurst_label, vr_label, final_label, agreement = classify_regime(hurst, vr)
    
    results.append({
        'day': day,
        'hurst_exponent': hurst,
        'hurst_classification': hurst_label,
        'variance_ratio_q5': vr,
        'vr_z_statistic': vr_z,
        'vr_classification': vr_label,
        'tests_agreement': agreement,
        'final_regime': final_label,
    })

regime_df = pd.DataFrame(results)

summary = regime_df['final_regime'].value_counts().reset_index()
summary.columns = ['regime', 'num_days']
summary['percentage'] = (summary['num_days'] / len(regime_df) * 100).round(1)

agreement_rate = (regime_df['tests_agreement'] == 'Agree').mean() * 100

summary_row = pd.DataFrame([{
    'regime': 'TOTAL',
    'num_days': len(regime_df),
    'percentage': 100.0
}])
summary = pd.concat([summary, summary_row], ignore_index=True)

regime_df.to_csv("results/regime_classification_85days.csv", index=False)
summary.to_csv("results/regime_summary_breakdown.csv", index=False)
