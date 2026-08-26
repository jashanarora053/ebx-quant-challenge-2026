# EBX Quantitative Research Challenge — Prepathon 2026

> **Statistical Foundations & Regime Dynamics (Parts 1–3)**  
> High-frequency tick-level price analysis across 70 trading days: Data hygiene, microstructure diagnostics, tail risk, and regime classification.

---

## 📌 Project Overview

This repository contains an end-to-end quantitative research pipeline built for the **EBX Quant Challenge (Inter-IIT Prepathon 2026)**. The analysis evaluates 1-second price bars across three core dimensions:
1. **Part 1 (Data Hygiene & Descriptive Statistics):** Ingestion sanity checks, microstructure noise diagnostics via autocorrelation, and 5-minute binned intraday volatility seasonality.
2. **Part 2 (Distributional & Tail Risk Analysis):** Formal normality testing (JB, D'Agostino, AD, Shapiro-Wilk), empirical vs. Gaussian $\sigma$-event multipliers, volatility clustering, non-parametric Hill tail indices, and a rare-event catalogue.
3. **Part 3 (Regime Classification & Dynamics):** Independent daily classification into Random Walk vs. Momentum regimes using Hurst Exponent and Variance Ratio tests, coupled with a first-order Markov transition probability matrix.

---

## 📁 Repository Structure

├── src/ # Source code modules │ ├── ingestion/ │ │ └── Ingestion_sanity.py # Part 1: Pipeline sanity checks │ ├── descriptive_stats.py # Part 1: Per-day & pooled statistics │ ├── intraday_seasonality.py # Part 1: 5-min binned intraday volatility │ ├── normality_testing.py # Part 2: JB, D'Agostino, AD, and Shapiro tests │ ├── sigma_analysis.py # Part 2: Sigma-event exceedance calculations │ ├── volatility_clustering.py # Part 2: Runs test & clustering analysis │ ├── hill_estimator.py # Part 2: Hill tail index & excess kurtosis │ ├── rare_event_catalogue.py # Part 2: Top 20 extreme 1-minute moves │ ├── regime_classification.py # Part 3: Hurst exponent & Variance Ratio │ └── regime_transitions.py # Part 3: Markov transition matrix │ ├── results/ # Generated CSV artifacts │ ├── sanity_report.csv │ ├── per_day_descriptive_stats.csv │ ├── pooled_descriptive_stats.csv │ ├── acf_returns_lag1_to_60.csv │ ├── volatility_seasonality.csv │ ├── normality_tests.csv │ ├── sigma_events_per_day.csv │ ├── sigma_events_pooled.csv │ ├── sigma_clustering_per_day.csv │ ├── sigma_clustering_test.csv │ ├── hill_tail_index.csv │ ├── rare_event_catalogue.csv │ ├── regime_classification_85days.csv │ ├── regime_summary_breakdown.csv │ ├── regime_transition_counts.csv │ └── regime_transition_matrix.csv │ ├── plots/ # High-resolution figures │ ├── volatility_seasonality.png │ ├── 1m_return_vs_fitted_normal_curve_plot.png │ ├── 5m_return_vs_fitted_normal_curve_plot.png │ ├── 1m_qq_plot_hill_tail_index.png │ ├── 5m_qq_plot_hill_tail_index.png │ ├── volatility_clustering_bar_chart.png │ └── regime_transition_heatmap.png │ ├── reports/ # Final research submission │ ├── final_report.tex │ └── final_report.pdf │ └── README.mds
---

## 🔬 Key Empirical Findings

### Part 1: Data Hygiene & Descriptive Statistics
* **Data Scope:** 70 available sessions (Days 1–64, 80–85) analyzed. Days 65–79 are explicitly treated as an un-interpolated missing gap.
* **Integrity:** Zero malformed timestamps, zero missing seconds, and zero corrupt prices. Five sessions (Days 60–64) are truncated ($\approx$ 4h long) and analyzed at their natural length.
* **Microstructure Noise:** Lag-1 autocorrelation of 1-second returns is $\hat{\rho}(1) = -0.0171$ (decaying to zero by lag 5), confirming bid-ask bounce is negligible at 1-minute and 5-minute sampling grids.
* **Intraday Seasonality:** Realized volatility exhibits a distinct **U-shaped profile**—peaking at the open ($4.62 \times 10^{-5}$), dropping by 56% to a midday trough near 02:45 ($2.03 \times 10^{-5}$), and ramping into the close ($3.46 \times 10^{-5}$).

### Part 2: Distributional & Tail Risk Analysis
* **Normality Rejected:** All four goodness-of-fit tests (Jarque–Bera, D'Agostino $K^2$, Anderson–Darling, and Shapiro–Wilk) reject Gaussianity across all scopes ($p \approx 0.00$).
* **Severe Leptokurtosis:** Pooled excess kurtosis is **+16.29** for 1-minute and **+11.43** for 5-minute returns.
* **Gaussian VaR Breakdown:** $3\sigma+$ moves occur **$6.21\times$** more often than Gaussian expectations, $4\sigma$ moves occur **$106\times$** more often, and $5\sigma$ events occur over **$5,000\times$** more frequently (5,014 actual vs. $<1$ expected).
* **Tail Heaviness:** Non-parametric Hill tail indices ($\hat{\alpha}_{\text{1m}} = 4.02$, $\hat{\alpha}_{\text{5m}} = 5.16$) confirm heavy power-law tails that diverge from the Gaussian limit ($\alpha \to \infty$).
* **Volatility Clustering:** Shocks concentrate on specific days (e.g., Day 14 produces $\approx 600$ $3\sigma+$ events, $>2.4\times$ the daily median of 243.5).
* **Rare Events:** Top 20 extreme moves cluster into 4 localized flash windows, led by Day 36 (+0.785% jump after a 118-tick liquidity vacuum).

### Part 3: Regime Classification & Markov Dynamics
* **Regime Breakdown:** Using pre-declared thresholds on the Hurst Exponent ($H$) and Variance Ratio ($VR(5)$), **87.1%** of sessions classify as **Random Walk** and **12.9%** as **Momentum** (0% Mean-Reverting).
* **Markov Persistence:** Random Walk sessions exhibit an **88.3%** same-regime continuation probability, while Momentum days act as transient 1-day bursts (77.8% probability of reverting to Random Walk).


---
# Part 1: Ingestion & Descriptive Statistics
python src/ingestion/Ingestion_sanity.py
python src/descriptive_stats.py
python src/intraday_seasonality.py

# Part 2: Normality, Tail Risk & Rare Events
python src/normality_testing.py
python src/sigma_analysis.py
python src/volatility_clustering.py
python src/hill_estimator.py
python src/rare_event_catalogue.py

# Part 3: Regime Classification & Markov Transitions
python src/regime_classification.py
python src/regime_transitions.py

---

## 🛠️ Tech Stack & Dependencies

| Tool / Library | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **pandas** | Data manipulation, tick resampling, and CSV I/O |
| **numpy** | Vectorized array operations and numerical math |
| **scipy.stats** | Statistical hypothesis tests (JB, D'Agostino, AD, Shapiro-Wilk, Runs test) |
| **matplotlib / seaborn** | High-resolution publication-quality plots |
| **LaTeX (Overleaf / pdflatex)** | Formal academic research report typesetting |

---

## 📄 License & Attribution

This repository is submitted as part of the **EBX Quantitative Data Challenge (Prepathon 2026)** .
