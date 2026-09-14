import numpy as np
import pandas as pd
import os
import glob
import re
from scipy.stats import spearmanr

# Forward return horizons (seconds)
hor = [10, 30, 60, 120, 300]


def fwd_ret(day_df):
    #Calculate forward N-second returns and drop the trailing NaNs
    price = day_df["Price"]
    for n in hor:
        future_price = price.shift(-n)
        day_df[f"fwd_ret_{n}"] = (future_price - price) / price

    day_df = day_df.dropna(subset=[f"fwd_ret_{n}" for n in hor])
    return day_df


def calc_ic(data_folder="data", save_dir="results"):

    # Load valid file paths up to Day 85 in numeric order
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
        print(f"Error: No data files found in '{data_folder}'")
        return

    # Store processed days and concatenate them into a single dataframe
    chunks = []  
    print(f"Processing {len(day_files)} days of data...")
    
    for idx, fpath in enumerate(day_files):
        try:
            raw = pd.read_csv(fpath)
            if "Price" not in raw.columns:
                continue
            day_with_fwd = fwd_ret(raw)
            day_with_fwd = day_with_fwd.dropna()
            chunks.append(day_with_fwd)
        except Exception:
            continue

    if not chunks:
        print("Error: No valid chunks processed.")
        return

    combined = pd.concat(chunks, ignore_index=True)

    # Identify masked feature columns
    skip = {"Time", "Price"}
    fwd_cols = [f"fwd_ret_{n}" for n in hor]
    feature_cols = [c for c in combined.columns if c not in skip and c not in fwd_cols]
    
    # Compute IC (Pearson + Spearman) for every feature x horizon
    print("Calculating Information Coefficients...")
    rows = []
    for feat in feature_cols:
        if combined[feat].isna().all():
            continue

        feat_values = combined[feat]
        
        for n in hor:
            fwd = combined[f"fwd_ret_{n}"]

            # Data filtering: eliminate NaN values pairwise
            mask = feat_values.notna() & fwd.notna()
            x = feat_values[mask]
            y = fwd[mask]

            if len(x) < 100:
                continue

            pearson_ic = x.corr(y)
            spearman_ic, _ = spearmanr(x, y)

            rows.append({
                "feature": feat,
                "family": feat.split("_")[0],
                "forward_horizon": n,
                "pearson_ic": round(pearson_ic, 6),
                "spearman_ic": round(spearman_ic, 6),
                "abs_pearson_ic": round(abs(pearson_ic), 6),
                "abs_spearman_ic": round(abs(spearman_ic), 6),
            })

    ic_df = pd.DataFrame(rows)

    # Aggregate features (preserving the original CSV column order)
    ranking = (
        ic_df.groupby("feature", sort=False)
        .agg(
            family=("family", "first"),
            avg_abs_pearson=("abs_pearson_ic", "mean"),
            avg_abs_spearman=("abs_spearman_ic", "mean"),
            max_abs_spearman=("abs_spearman_ic", "max"),
            best_horizon=("abs_spearman_ic", "idxmax"),
        )
        .reset_index()
    )

    # Map the best_horizon idxmax back to the actual time value
    ranking["best_horizon"] = ranking["best_horizon"].map(
        lambda idx: ic_df.loc[idx, "forward_horizon"] if idx in ic_df.index else np.nan
    )

    # Add a predictive rank column (1 = highest predictive power)
    ranking["predictive_rank"] = ranking["avg_abs_spearman"].rank(ascending=False, method="min").astype(int)
    
    cols = ["predictive_rank", "feature", "family", "avg_abs_pearson", "avg_abs_spearman", "max_abs_spearman", "best_horizon"]
    ranking = ranking[cols]

    # Family-level summary (Sorted by strongest families)
    family_summary = (
        ic_df.groupby("family")
        .agg(
            avg_abs_spearman=("abs_spearman_ic", "mean"),
            max_abs_spearman=("abs_spearman_ic", "max"),
            n_features=("feature", "nunique"),
        )
        .sort_values("avg_abs_spearman", ascending=False)
        .reset_index()
    )

    os.makedirs(save_dir, exist_ok=True)

    # Save to CSV
    ic_df.to_csv(os.path.join(save_dir, "predictive_ic_full.csv"), index=False)
    ranking.to_csv(os.path.join(save_dir, "predictive_ic_ranking.csv"), index=False)
    family_summary.to_csv(os.path.join(save_dir, "predictive_ic_family_summary.csv"), index=False)
    
    print(f"Results successfully saved to '{save_dir}/'")


if __name__ == "__main__":
    # Standard repository structure assuming data is in a 'data' folder
    calc_ic(data_folder="data", save_dir="results")
