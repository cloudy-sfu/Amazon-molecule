from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import wilcoxon
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from argparse import ArgumentParser

# %% Constants.
parser = ArgumentParser()
parser.add_argument('--lag', type=int)
parser.add_argument('--dataset_name', type=str)
command, _ = parser.parse_known_args()
lag = command.lag
dataset_name = command.dataset_name

# %% Load data.
_, ts_test = pd.read_pickle(f'raw/1_{dataset_name}_std.pkl')

# %% Split by mass.
cols = ts_test.columns
mass_heights = map(lambda x: x.split('_') + [x], cols)
cols_grouped_mass = defaultdict(list)
all_possible_heights = []
for mass, height, mass_height in mass_heights:
    cols_grouped_mass[mass].append(mass_height)
    all_possible_heights.append(height)
all_possible_heights = np.unique(all_possible_heights)
heights_pair = list((fi, fj) for fi, fj in product(all_possible_heights, repeat=2) if fi != fj)

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
w_val = pd.DataFrame(index=cols_grouped_mass.keys(), columns=pd.MultiIndex.from_tuples(heights_pair), dtype=float)
p_val = deepcopy(w_val)
pbar = tqdm(total=sum(len(x) ** 2 for x in cols_grouped_mass.values()) - len(cols))

@np.vectorize
def decimal_non_zero(x):
    return format(x, '.2f').removeprefix('0')

# %% Infer causal network per mass.
for mass, cols_this_mass in cols_grouped_mass.items():
    x_test, y_test = moving_window(ts_test[cols_this_mass].values, k=lag)
    cols_height = [col.split('_')[1] for col in cols_this_mass]

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
        try:
            ur = tf.keras.models.load_model(f'raw/9_{dataset_name}_nn_{lag}/{mass}_{i}_{j}_ur.h5')
            r = tf.keras.models.load_model(f'raw/9_{dataset_name}_nn_{lag}/{mass}_{i}_{j}_r.h5')
        except OSError:
            w_val.loc[mass, (cols_height[i], cols_height[j])] = np.nan
            p_val.loc[mass, (cols_height[i], cols_height[j])] = np.nan
            print(f'[Warning] {mass}, {cols_height[i]}, {cols_height[j]} models are broken.')
            pbar.update(1)
            continue

        # Use model to predict
        y_test_ur_hat = ur.predict(x_test_ur, batch_size=x_test.shape[0], verbose=0)
        y_test_r_hat = r.predict(x_test_r, batch_size=x_test.shape[0], verbose=0)

        # Infer causality
        err_ur = (y_test_ur_hat.flatten() - y_test_ur) ** 2
        err_r = (y_test_r_hat.flatten() - y_test_r) ** 2
        wilcoxon_results = wilcoxon(x=err_r, y=err_ur, method='approx', alternative='greater')
        w_val.loc[mass, (cols_height[i], cols_height[j])] = wilcoxon_results.statistic
        p_val.loc[mass, (cols_height[i], cols_height[j])] = wilcoxon_results.pvalue
        pbar.update(1)

# %% Heatmap
fig, ax = plt.subplots(figsize=(0.75 * p_val.shape[1], 0.5 * p_val.shape[0] + 1))
heatmap = sns.heatmap(p_val.values, square=True, linewidths=.5, cmap='coolwarm', vmin=0, vmax=0.1,
                      annot=decimal_non_zero(p_val.values), fmt='', ax=ax)
ax.set_ylabel('Mass')
ax.set_xlabel('(Cause, Effect)')
ax.set_xticklabels(p_val.columns, rotation=45)
ax.set_yticklabels(p_val.index, rotation=0)
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
fig.savefig(f'results/10_{dataset_name}_p_{lag}.eps')
plt.close(fig)

# %% Export.
pd.to_pickle(w_val, f'raw/10_{dataset_name}_w_{lag}.pkl')
pd.to_pickle(p_val, f'raw/10_{dataset_name}_p_{lag}.pkl')
