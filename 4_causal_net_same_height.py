import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict

# %% Constant.
dataset_name = 'default_1h'
lag = 12

# %% Load data.
p_val = pd.read_pickle(f'raw/3_{dataset_name}_p_{lag}.pkl')

# %% Initialization.
@np.vectorize
def decimal_non_zero(x):
    return format(x, '.2f').removeprefix('0')

# %% Get column names.
ts_train_valid, ts_test = pd.read_pickle(f'raw/1_{dataset_name}_std.pkl')
cols = ts_train_valid.columns
mass_heights = map(lambda x: x.split('_') + [x], cols)
cols_grouped_height = defaultdict(list)
for mass, height, mass_height in mass_heights:
    cols_grouped_height[height].append(mass_height)

# %% Draw causality p-value matrix.
for height, p0 in p_val.items():
    fig, ax = plt.subplots(figsize=(7.5, 6))
    mask = np.zeros_like(p0, dtype=bool)
    mask[np.diag_indices_from(mask)] = True
    heatmap = sns.heatmap(p0, mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                          vmin=0, vmax=0.1, annot=decimal_non_zero(p0), fmt='', ax=ax)
    ax.set_ylabel('Cause')
    ax.set_xlabel('Effect')
    ax.set_xticklabels(cols_grouped_height[height], rotation=45)
    ax.set_yticklabels(cols_grouped_height[height], rotation=0)
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.10, right=1)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    fig.savefig(f'results/4_{dataset_name}_p_{lag}_{height}.eps')
    plt.close(fig)
