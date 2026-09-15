import numpy as np
import pandas as pd
import os
import glob
import re
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
 
# Cumulative-explained-variance cut-offs to report
VAR_THRESHOLDS = (0.90, 0.95, 0.99)
 
# Features closer than this in correlation distance (1 - |rho|) are grouped.
# 0.30 => any pair with |rho| > 0.70 lands in the same cluster.
CLUSTER_DIST = 0.30
 
# Number of PCs to export loadings / communalities for
N_LOADINGS = 6
 
 

# Loading ----------------------------------------------------------------------
def load_day_paths(data_folder="data", max_day=85):
    raw_files = glob.glob(os.path.join(data_folder, "Day*.csv"))
    valid_files = []
    for f in raw_files:
        filename = os.path.basename(f)
        match = re.search(r'Day_?(\d+)\.csv', filename, re.IGNORECASE)
        if match:
            day_num = int(match.group(1))
            if day_num <= max_day:
                valid_files.append((day_num, f))
 
    valid_files.sort(key=lambda x: x[0])
    return valid_files
 
 
def load_features(data_folder="data", max_day=85):
    """Concatenate all days, returning the feature block and a day-id column."""
    valid_files = load_day_paths(data_folder, max_day)
    if not valid_files:
        print(f"Error: No data files found in '{data_folder}'")
        return None, None
 
    skip = {"Time", "Price"}
    chunks, day_ids = [], []
 
    print(f"Processing {len(valid_files)} days of data...")
    for day_num, fpath in valid_files:
        try:
            raw = pd.read_csv(fpath)
        except Exception:
            continue
 
        # Drop identifiers and anything left over from the 4.2 forward-return step
        cols = [
            c for c in raw.columns
            if c not in skip and not c.startswith("fwd_ret")
        ]
        if not cols:
            continue
 
        block = raw[cols].apply(pd.to_numeric, errors="coerce")
        block = block.dropna()
        if block.empty:
            continue
 
        chunks.append(block)
        day_ids.append(pd.Series(day_num, index=block.index))
 
    if not chunks:
        print("Error: No valid chunks processed.")
        return None, None
 
    combined = pd.concat(chunks, ignore_index=True)
    days = pd.concat(day_ids, ignore_index=True)
 
    # Zero-variance columns make the correlation matrix undefined
    stds = combined.std()
    dead = stds[(stds == 0) | stds.isna()].index.tolist()
    if dead:
        print(f"Warning: dropping {len(dead)} constant/degenerate features: {dead}")
        combined = combined.drop(columns=dead)
 
    return combined, days
 
 
 
# Effective sample size and the noise floor -----------------------------------
def effective_sample_size(df, days):
    """
    Deflate the row count by an AR(1) autocorrelation correction:
        T_eff = T * (1 - rho) / (1 + rho)
    rho is estimated per feature *within* each day (so day boundaries don't
    leak), then averaged. Overlapping rolling windows push rho towards 1, which
    is precisely why the naive T is misleading.
    """
    rhos = []
    for _, idx in df.groupby(days.values).groups.items():
        sub = df.loc[idx]
        if len(sub) < 50:
            continue
        for c in sub.columns:
            s = sub[c].values
            r = np.corrcoef(s[:-1], s[1:])[0, 1]
            if np.isfinite(r):
                rhos.append(r)
 
    if not rhos:
        return float(len(df)), np.nan
 
    rho_bar = float(np.clip(np.mean(rhos), 0.0, 0.9999))
    t_eff = len(df) * (1.0 - rho_bar) / (1.0 + rho_bar)
    return max(t_eff, 2.0), rho_bar
 
 
def mp_upper_edge(p, t_eff):
    """
    Marchenko-Pastur bulk edge for a pure-noise correlation matrix:
        lambda_+ = (1 + sqrt(q))^2,  q = p / T_eff
    Eigenvalues below this are statistically indistinguishable from noise.
    """
    q = p / float(t_eff)
    return (1.0 + np.sqrt(q)) ** 2, q
 
 
# PCA diagnostics
def eigen_decompose(corr):
    """Eigendecomposition of a correlation matrix, sorted descending."""
    vals, vecs = np.linalg.eigh(corr.values)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]
 
 
def effective_dimension_metrics(eigvals, mp_edge=None):
    """
    Several ways to turn a spectrum into 'how many dimensions'. They disagree by
    design; reporting a range is the honest answer.
    """
    lam = np.clip(eigvals, 0.0, None)
    p = len(lam)
    total = lam.sum()
    ratio = lam / total
    cum = np.cumsum(ratio)
 
    pos = lam[lam > 1e-12]
    pr = ratio[ratio > 1e-12]
 
    metrics = {
        "n_features": p,
        # Directions carrying more variance than one standardised feature alone
        "kaiser_count_lambda_gt_1": int((lam > 1.0).sum()),
        # Threshold-free: heavily weights the dominant directions
        "participation_ratio": float((lam.sum() ** 2) / np.square(lam).sum()),
        # Threshold-free: exp(Shannon entropy of the normalised spectrum)
        "entropy_effective_rank": float(np.exp(-np.sum(pr * np.log(pr)))),
        # trace / lambda_max -- how many "lambda_max-sized" directions fit
        "stable_rank": float(total / lam[0]),
        "lambda_max": float(lam[0]),
        "lambda_min": float(lam[-1]),
        "condition_number": float(lam[0] / pos[-1]) if len(pos) else np.inf,
        "numerical_rank_tol_1e-10": int((lam > 1e-10 * lam[0]).sum()),
    }
 
    for thr in VAR_THRESHOLDS:
        k = int(np.searchsorted(cum, thr) + 1)
        metrics[f"k_for_{int(thr * 100)}pct_variance"] = min(k, p)
 
    if mp_edge is not None and np.isfinite(mp_edge):
        metrics["mp_upper_edge"] = float(mp_edge)
        metrics["n_signal_eigenvalues_above_mp"] = int((lam > mp_edge).sum())
 
    return metrics, ratio, cum
 
 
def vif_table(corr):
    """
    VIF_i = (C^-1)_ii, and R^2_i = 1 - 1/VIF_i is the share of feature i's
    variance explained by a linear combination of every other feature. This is
    the direct measure of multi-way redundancy that pairwise correlation misses.
    """
    try:
        inv = np.linalg.inv(corr.values)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr.values)
 
    vif = np.clip(np.diag(inv), 1.0, None)
    r2 = 1.0 - 1.0 / vif
 
    return pd.DataFrame({
        "feature": corr.columns,
        "family": [str(c).split("_")[0] for c in corr.columns],
        "vif": np.round(vif, 4),
        "r2_vs_all_others": np.round(r2, 6),
    }).sort_values("r2_vs_all_others", ascending=False).reset_index(drop=True)
 
 
def cluster_features(corr):
    """
    Hierarchical clustering on correlation distance (1 - |rho|). The cluster
    count at a fixed threshold is an intuitive, non-spectral second opinion on
    effective dimensionality, and the ordering makes the heatmap block structure
    readable.
    """
    d = 1.0 - corr.abs().values
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2.0, 0.0, None)
 
    link = linkage(squareform(d, checks=False), method="average")
    order = leaves_list(link)
    labels = fcluster(link, t=CLUSTER_DIST, criterion="distance")
 
    abs_corr = corr.abs().copy()
    np.fill_diagonal(abs_corr.values, np.nan)
 
    table = pd.DataFrame({
        "feature": corr.columns,
        "family": [str(c).split("_")[0] for c in corr.columns],
        "cluster_id": labels,
        "max_abs_corr_other": abs_corr.max(axis=1).round(6).values,
        "closest_partner": abs_corr.idxmax(axis=1).values,
    })
    table = table.sort_values(["cluster_id", "feature"]).reset_index(drop=True)
 
    ordered_cols = [corr.columns[i] for i in order]
    return table, ordered_cols, int(labels.max())
 
 
def daily_stability(df, days):
    """Recompute effective rank per day -- correlation structure drifts."""
    rows = []
    for day, idx in df.groupby(days.values).groups.items():
        sub = df.loc[idx]
        if len(sub) < 10 * sub.shape[1]:
            continue
        c = sub.corr(method="pearson").dropna(how="all").dropna(axis=1, how="all")
        if c.shape[0] < 2:
            continue
        vals, _ = eigen_decompose(c)
        m, _, _ = effective_dimension_metrics(vals)
        rows.append({
            "day": int(day),
            "n_rows": len(sub),
            "entropy_effective_rank": round(m["entropy_effective_rank"], 4),
            "participation_ratio": round(m["participation_ratio"], 4),
            "k_for_95pct_variance": m["k_for_95pct_variance"],
            "kaiser_count_lambda_gt_1": m["kaiser_count_lambda_gt_1"],
            "pc1_explained_var": round(m["lambda_max"] / m["n_features"], 6),
        })
    return pd.DataFrame(rows)
 
 
# Optional charts 
def make_charts(corr, ordered_cols, eigvals, ratio, cum, vecs, feature_names,
                daily, mp_edge, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable - skipping charts.")
        return
 
    # Clustered correlation heatmap
    cm = corr.loc[ordered_cols, ordered_cols]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(ordered_cols)))
    ax.set_xticklabels(ordered_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ordered_cols)))
    ax.set_yticklabels(ordered_cols, fontsize=7)
    ax.set_title("Feature correlation matrix (hierarchically ordered)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "redundancy_corr_heatmap.png"), dpi=150)
    plt.close(fig)
 
    # Scree + cumulative variance
    ks = np.arange(1, len(eigvals) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.bar(ks, eigvals, color="#4C72B0")
    a1.axhline(1.0, ls="--", c="grey", label="Kaiser (lambda = 1)")
    if np.isfinite(mp_edge):
        a1.axhline(mp_edge, ls=":", c="crimson",
                   label=f"MP noise floor = {mp_edge:.2f}")
    a1.set_xlabel("Principal component")
    a1.set_ylabel("Eigenvalue")
    a1.set_title("Scree plot")
    a1.legend(fontsize=8)
 
    a2.plot(ks, cum, marker="o", color="#55A868")
    for thr in VAR_THRESHOLDS:
        a2.axhline(thr, ls="--", lw=0.8, c="grey")
        a2.text(len(ks), thr, f" {int(thr*100)}%", va="center", fontsize=8)
    a2.set_xlabel("Number of components")
    a2.set_ylabel("Cumulative explained variance")
    a2.set_title("Cumulative variance")
    a2.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "redundancy_scree.png"), dpi=150)
    plt.close(fig)
 
    # Loadings heatmap for the leading PCs
    k = min(N_LOADINGS, vecs.shape[1])
    fig, ax = plt.subplots(figsize=(1.4 * k + 4, 0.35 * len(feature_names) + 2))
    im = ax.imshow(vecs[:, :k], vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"PC{i+1}\n{ratio[i]*100:.1f}%" for i in range(k)], fontsize=8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title("Eigenvector loadings")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "redundancy_loadings.png"), dpi=150)
    plt.close(fig)
 
    # Day-by-day stability
    if not daily.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily["day"], daily["entropy_effective_rank"],
                marker=".", label="Entropy effective rank")
        ax.plot(daily["day"], daily["participation_ratio"],
                marker=".", label="Participation ratio")
        ax.plot(daily["day"], daily["k_for_95pct_variance"],
                marker=".", label="k for 95% variance")
        ax.set_xlabel("Day")
        ax.set_ylabel("Effective dimensions")
        ax.set_title("Stability of effective dimensionality across days")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "redundancy_daily_stability.png"), dpi=150)
        plt.close(fig)
 
 
# Main
def calc_redundancy(data_folder="data", save_dir="results", max_day=85,
                    make_plots=True):
    combined, days = load_features(data_folder, max_day)
    if combined is None:
        return
 
    feature_names = list(combined.columns)
    p = len(feature_names)
    n = len(combined)
    print(f"Loaded {n:,} rows x {p} features.")
 
    # --- Correlation matrices -----------------------------------------
    print("Computing correlation matrices...")
    corr_p = combined.corr(method="pearson")
    rho_s, _ = spearmanr(combined.values)
    if np.ndim(rho_s) == 0:
        rho_s = np.array([[1.0]])
    corr_s = pd.DataFrame(rho_s, index=feature_names, columns=feature_names)
 
    # --- Effective sample size and noise floor ------------------------
    t_eff, rho_bar = effective_sample_size(combined, days)
    mp_edge, q = mp_upper_edge(p, t_eff)
    print(f"Mean lag-1 autocorrelation: {rho_bar:.4f}")
    print(f"Effective sample size: {t_eff:,.0f} (vs {n:,} raw rows), q = {q:.4f}")
    print(f"Marchenko-Pastur noise floor: lambda_+ = {mp_edge:.4f}")
 
    # --- PCA ----------------------------------------------------------
    print("Running PCA on the correlation matrix...")
    eigvals, vecs = eigen_decompose(corr_p)
    metrics, ratio, cum = effective_dimension_metrics(eigvals, mp_edge)
 
    # Communality: share of each feature's variance captured by the top-k PCs.
    # For a correlation matrix, sum_j V_ij^2 * lambda_j = 1 exactly.
    k_comm = min(N_LOADINGS, p)
    comm = (np.square(vecs[:, :k_comm]) * eigvals[:k_comm]).cumsum(axis=1)
 
    eig_df = pd.DataFrame({
        "pc": np.arange(1, p + 1),
        "eigenvalue": np.round(eigvals, 6),
        "explained_var": np.round(ratio, 6),
        "cumulative_var": np.round(cum, 6),
        "above_kaiser_1": eigvals > 1.0,
        "above_mp_noise_floor": eigvals > mp_edge,
    })
 
    load_df = pd.DataFrame(
        np.round(vecs[:, :k_comm], 6),
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(k_comm)],
    )
    load_df.insert(0, "family", [str(c).split("_")[0] for c in feature_names])
    for i in range(k_comm):
        load_df[f"communality_top{i+1}"] = np.round(comm[:, i], 6)
    load_df = load_df.reset_index().rename(columns={"index": "feature"})
 
    # --- Redundancy views --------------------------------------------
    vifs = vif_table(corr_p)
    clusters, ordered_cols, n_clusters = cluster_features(corr_p)
    metrics["n_clusters_at_abs_corr_0.70"] = n_clusters
    metrics["mean_lag1_autocorr"] = round(rho_bar, 6)
    metrics["n_rows_raw"] = n
    metrics["n_rows_effective"] = round(t_eff, 2)
    metrics["q_ratio_p_over_t_eff"] = round(q, 6)
    metrics["mean_abs_offdiag_corr"] = round(
        float(corr_p.abs().values[~np.eye(p, dtype=bool)].mean()), 6
    )
    metrics["max_vif"] = float(vifs["vif"].max())
    metrics["n_features_r2_above_0.99"] = int((vifs["r2_vs_all_others"] > 0.99).sum())
 
    # --- Stability ----------------------------------------------------
    print("Checking day-by-day stability...")
    daily = daily_stability(combined, days)
    if not daily.empty:
        metrics["daily_entropy_rank_mean"] = round(
            float(daily["entropy_effective_rank"].mean()), 4)
        metrics["daily_entropy_rank_std"] = round(
            float(daily["entropy_effective_rank"].std()), 4)
        metrics["daily_entropy_rank_min"] = round(
            float(daily["entropy_effective_rank"].min()), 4)
        metrics["daily_entropy_rank_max"] = round(
            float(daily["entropy_effective_rank"].max()), 4)
 
    summary = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    )
 
    # --- Save ---------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    corr_p.round(6).to_csv(os.path.join(save_dir, "redundancy_corr_pearson.csv"))
    corr_s.round(6).to_csv(os.path.join(save_dir, "redundancy_corr_spearman.csv"))
    eig_df.to_csv(os.path.join(save_dir, "redundancy_eigenvalues.csv"), index=False)
    load_df.to_csv(os.path.join(save_dir, "redundancy_loadings.csv"), index=False)
    vifs.to_csv(os.path.join(save_dir, "redundancy_vif.csv"), index=False)
    clusters.to_csv(os.path.join(save_dir, "redundancy_clusters.csv"), index=False)
    summary.to_csv(os.path.join(save_dir, "redundancy_summary.csv"), index=False)
    if not daily.empty:
        daily.to_csv(os.path.join(save_dir, "redundancy_daily_stability.csv"),
                     index=False)
 
    if make_plots:
        make_charts(corr_p, ordered_cols, eigvals, ratio, cum, vecs,
                    feature_names, daily, mp_edge, save_dir)
 
    # --- Report -------------------------------------------------------
    print("\n" + "=" * 62)
    print("EFFECTIVE DIMENSIONALITY")
    print("=" * 62)
    print(f"  Nominal features                : {p}")
    print(f"  Eigenvalues > 1 (Kaiser)        : {metrics['kaiser_count_lambda_gt_1']}")
    print(f"  Eigenvalues > MP noise floor    : {metrics.get('n_signal_eigenvalues_above_mp', 'n/a')}")
    print(f"  Components for 90% variance     : {metrics['k_for_90pct_variance']}")
    print(f"  Components for 95% variance     : {metrics['k_for_95pct_variance']}")
    print(f"  Components for 99% variance     : {metrics['k_for_99pct_variance']}")
    print(f"  Participation ratio             : {metrics['participation_ratio']:.2f}")
    print(f"  Entropy effective rank          : {metrics['entropy_effective_rank']:.2f}")
    print(f"  Clusters at |rho| > 0.70        : {n_clusters}")
    print(f"  Condition number                : {metrics['condition_number']:.1f}")
    print(f"  PC1 alone explains              : {ratio[0]*100:.1f}% of variance")
    print(f"\nResults successfully saved to '{save_dir}/'")
 
 
if __name__ == "__main__":
    # Standard repository structure assuming data is in a 'data' folder
    calc_redundancy(data_folder="data", save_dir="results")
 