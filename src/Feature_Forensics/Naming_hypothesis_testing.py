import numpy as np
import pandas as pd
import os
import glob
from functools import partial
import re


def _log_ret(price, w=1):
    #Log return series
    return np.log(price / price.shift(w))


def _roll(series, w):
    #Rolling window with strict warm-up
    return series.rolling(w, min_periods=w)


FEATURE_BLUEPRINTS = {
    # Price Level Transforms
    "sma":            lambda p, w: _roll(p, w).mean(),
    "sma_median":     lambda p, w: _roll(p, w).median(),
    "sma_dev":        lambda p, w: p - _roll(p, w).mean(),
    "sma_pct_dev":    lambda p, w: p / _roll(p, w).mean() - 1.0,
    "exp_ma":         lambda p, w: p.ewm(span=w, min_periods=w, adjust=False).mean(),
    "price_std":      lambda p, w: _roll(p, w).std(ddof=1),
    "price_var":      lambda p, w: _roll(p, w).var(ddof=1),
    "range_low":      lambda p, w: _roll(p, w).min(),
    "range_high":     lambda p, w: _roll(p, w).max(),

    # Normalized 
    "z_norm":         lambda p, w: (p - _roll(p, w).mean()) / _roll(p, w).std(ddof=1),
    "pct_from_high":  lambda p, w: p / _roll(p, w).max() - 1.0,
    "pct_from_low":   lambda p, w: p / _roll(p, w).min() - 1.0,
    "stochastic":     lambda p, w: (p - _roll(p, w).min()) /
                                   (_roll(p, w).max() - _roll(p, w).min()).replace(0, np.nan),
    "pct_momentum":   lambda p, w: p / p.shift(w) - 1.0,

    # Log-Return Derived 
    "logret_avg":     lambda p, w: _roll(_log_ret(p), w).mean(),
    "logret_var":     lambda p, w: _roll(_log_ret(p), w).var(ddof=1),
    "logret_vol":     lambda p, w: _roll(_log_ret(p), w).std(ddof=1),
    "logret_absavg":  lambda p, w: _roll(_log_ret(p).abs(), w).mean(),
    "logret_downvol": lambda p, w: _roll(_log_ret(p).where(_log_ret(p) < 0, 0.0), w).std(ddof=1),
    "logret_upvol":   lambda p, w: _roll(_log_ret(p).where(_log_ret(p) > 0, 0.0), w).std(ddof=1),
}

# Extended time horizons (seconds) to catch T1 through T12
HORIZONS = [5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]


def build_candidate_matrix(day_df):
    #Attach all hand-built candidates to a single day's DataFrame

    price = day_df["Price"].copy()
    n = len(price)
    
    new_cols = {} 

    for h in HORIZONS:
        if h >= n:
            continue
        for tag, func in FEATURE_BLUEPRINTS.items():
            col = f"cand__{tag}__{h}"
            try:
                new_cols[col] = func(price, h)
            except Exception:
                new_cols[col] = np.nan

    candidates_df = pd.DataFrame(new_cols, index=day_df.index)
    return pd.concat([day_df, candidates_df], axis=1)


def run_hypothesis_test(data_folder, save_dir="results"):
    raw_files = glob.glob(os.path.join(data_folder, "Day*.csv"))

    valid_files = []
    for f in raw_files:
        filename = os.path.basename(f)
        match = re.search(r'Day_?(\d+)\.csv', filename, re.IGNORECASE)
        if match:
            day_num = int(match.group(1))
            if day_num <= 85: 
                valid_files.append((day_num, f))
    valid_files.sort(key=lambda x: x[0])
    
    day_files = [f[1] for f in valid_files]

    if not day_files:
        return

    chunks = []

    for idx, fpath in enumerate(day_files):
        try:
            raw = pd.read_csv(fpath)
            if "Price" not in raw.columns:
                continue

            enriched = build_candidate_matrix(raw)
            enriched = enriched.dropna()
            chunks.append(enriched)
        except Exception:
            continue

    if not chunks:
        return

    combined = pd.concat(chunks, ignore_index=True)

    # Separate masked vs candidate columns
    skip = {"Time", "Price"}
    target_cols = [c for c in combined.columns if c not in skip and not c.startswith("cand__")]
    cand_cols   = [c for c in combined.columns if c.startswith("cand__")]

    corr = combined[target_cols + cand_cols].corr(method="pearson")

    # Extract best matches
    rows = []

    for feat in target_cols:
        if combined[feat].isna().all():
            continue

        scores = corr.loc[feat, cand_cols].abs().sort_values(ascending=False)
        top1_name = scores.index[0]
        top1_r    = scores.iloc[0]
        top2_name = scores.index[1]
        top2_r    = scores.iloc[1]

        # Parse candidate name: cand__<formula>__<horizon>
        parts = top1_name.split("__")
        formula = parts[1] if len(parts) >= 3 else top1_name
        horizon = parts[2] if len(parts) >= 3 else "?"

        # Verdict
        if top1_r >= 0.95:
            tag = "EXACT MATCH"
        elif top1_r >= 0.70:
            tag = "STRONG MATCH"
        elif top1_r >= 0.30:
            tag = "WEAK"
        else:
            tag = "NO MATCH"

        accepted = top1_r >= 0.70

        # Family prefix: PB1, VB3, V2, etc.
        family = feat.split("_")[0]

        rows.append({
            "masked_col": feat,
            "family": family,
            "best_formula": formula,
            "best_window": horizon,
            "best_r": round(top1_r, 4),
            "runner_up_formula": "__".join(top2_name.split("__")[1:]),
            "runner_up_r": round(top2_r, 4),
            "verdict": tag,
            "price_derived": "YES" if accepted else "NO",
        })

    report = pd.DataFrame(rows)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "naming_hypothesis_results.csv")
    report.to_csv(out, index=False)


if __name__ == "__main__":
    run_hypothesis_test(r"C:\Users\omen\Downloads\archive")
