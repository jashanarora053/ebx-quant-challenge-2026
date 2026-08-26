import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("results", exist_ok=True)

regime_df = pd.read_csv("results/regime_classification_85days.csv")
regime_df = regime_df.sort_values('day').reset_index(drop=True)

today_regime = regime_df['final_regime'].iloc[:-1].values
tomorrow_regime = regime_df['final_regime'].iloc[1:].values

regimes = ['Mean-Reverting', 'Momentum', 'Random Walk']

transition_counts = pd.DataFrame(0, index=regimes, columns=regimes)

for t, tm in zip(today_regime, tomorrow_regime):
    if t in regimes and tm in regimes:
        transition_counts.loc[t, tm] += 1

row_sums = transition_counts.sum(axis=1)
transition_probs = transition_counts.div(row_sums, axis=0).round(3)
transition_probs = transition_probs.fillna(0)

transition_counts.index.name = 'from_regime'
transition_counts.columns.name = 'to_regime'
transition_probs.index.name = 'from_regime'
transition_probs.columns.name = 'to_regime'

transition_counts.to_csv("results/regime_transition_counts.csv")
transition_probs.to_csv("results/regime_transition_matrix.csv")
