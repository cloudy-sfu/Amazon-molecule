import os
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import wilcoxon
from tqdm import tqdm
from argparse import ArgumentParser
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

# %% Constants.
parser = ArgumentParser()
parser.add_argument('--lag', type=int)
parser.add_argument('--dataset_name', type=str)
command, _ = parser.parse_known_args()
lag = command.lag
dataset_name = command.dataset_name

# %% Load data.
_, ts_test = pd.read_pickle(f'raw/1_{dataset_name}_std.pkl')

# %% Split by height.
cols = ts_test.columns
mass_heights = map(lambda x: x.split('_') + [x], cols)
cols_grouped_height = defaultdict(list)
for mass, height, mass_height in mass_heights:
    cols_grouped_height[height].append(mass_height)

# %% Moving Window.
def moving_window(ts, k):
    """
    Make moving window samples from time series.
    :param ts: Time series.
    :param k: Length of the window.
    :return: x, y: fraction used as input, fraction used as output.
    """
    length = ts.shape[0]
    y = ts[k:]
    indices = np.tile(np.arange(k), [length - k, 1]) + np.arange(length - k)[:, np.newaxis]
    x = ts[indices]
    return x, y

# %% Initialization.
rng = np.random.RandomState(6611)
os.makedirs(f'results/7_{dataset_name}_p/', exist_ok=True)
w_val = {height: np.full(shape=(len(x), len(x)), fill_value=np.nan)
         for height, x in cols_grouped_height.items()}
p_val = deepcopy(w_val)
pbar = tqdm(total=sum(len(x)**2 for x in cols_grouped_height.values()) - len(cols))

@np.vectorize
def decimal_non_zero(x):
    return format(x, '.2f').removeprefix('0')

# %% Infer causal network per height.
for height, cols_this_height in cols_grouped_height.items():
    x_test, y_test = moving_window(ts_test[cols_this_height].values, k=lag)

    for i, j in product(range(x_test.shape[2]), repeat=2):
        if i == j:
            continue

        # Make the dataset
        x_test_ur = x_test
        y_test_ur = y_test[:, j]
        x_test_r = x_test.copy()
        rng.shuffle(x_test_r[:, :, i])
        y_test_r = y_test_ur

        # Load model
        ur = tf.keras.models.load_model(f'raw/3_{dataset_name}_nn_{lag}/{height}_{i}_{j}_ur.h5')
        r = tf.keras.models.load_model(f'raw/3_{dataset_name}_nn_{lag}/{height}_{i}_{j}_r.h5')

        # Use model to predict
        y_test_ur_hat = ur.predict(x_test_ur, batch_size=x_test.shape[0], verbose=0)
        y_test_r_hat = r.predict(x_test_r, batch_size=x_test.shape[0], verbose=0)

        # Infer causality
        err_ur = (y_test_ur_hat.flatten() - y_test_ur) ** 2
        err_r = (y_test_r_hat.flatten() - y_test_r) ** 2
        wilcoxon_results = wilcoxon(x=err_r, y=err_ur, method='approx', alternative='greater')
        w_val[height][i, j] = wilcoxon_results.statistic
        p_val[height][i, j] = wilcoxon_results.pvalue
        pbar.update(1)

    # Heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6))
    mask = np.zeros_like(p_val[height], dtype=bool)
    mask[np.diag_indices_from(mask)] = True
    heatmap = sns.heatmap(p_val[height], mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                          vmin=0, vmax=0.1, annot=decimal_non_zero(p_val[height]), fmt='', ax=ax)
    ax.set_ylabel('Cause')
    ax.set_xlabel('Effect')
    ax.set_xticklabels(cols_grouped_height[height], rotation=45)
    ax.set_yticklabels(cols_grouped_height[height], rotation=0)
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.10, right=1)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    fig.savefig(f'results/7_{dataset_name}_p/{lag}_{height}.eps')
    plt.close(fig)

# %% Export.
pd.to_pickle(w_val, f'raw/7_{dataset_name}_w_{lag}.pkl')
pd.to_pickle(p_val, f'raw/7_{dataset_name}_p_{lag}.pkl')
