from collections import defaultdict
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
from itertools import product
from mle import *
import joblib

# %% Constants.
lag = 96
dataset_name = 'default_15min'
n_boot = 200  # number of bootstrapping when estimating 'ncx2' dof.

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
p_val = pd.DataFrame(index=cols_grouped_mass.keys(), columns=pd.MultiIndex.from_tuples(heights_pair), dtype=float)
pbar = tqdm(total=sum(len(x) ** 2 for x in cols_grouped_mass.values()) - len(cols))
test_size = ts_test.shape[0] - lag

@np.vectorize
def decimal_non_zero(x):
    return format(x, '.2f').removeprefix('0')

def bootstrap_estimate_dof(errors, n):
    err_ur_boot = rng.choice(errors, size=n, replace=True)
    return estimate(err_ur_boot)['param'][0]


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
        ur = tf.keras.models.load_model(f'raw/9_{dataset_name}_nn_{lag}/{mass}_{i}_{j}_ur.h5')
        r = tf.keras.models.load_model(f'raw/9_{dataset_name}_nn_{lag}/{mass}_{i}_{j}_r.h5')

        # Use model to predict
        y_test_ur_hat = ur.predict(x_test_ur, batch_size=x_test.shape[0], verbose=0)
        y_test_r_hat = r.predict(x_test_r, batch_size=x_test.shape[0], verbose=0)

        # Infer causality
        err_ur = (y_test_ur_hat.flatten() - y_test_ur) ** 2
        err_r = (y_test_r_hat.flatten() - y_test_r) ** 2
        dof_ur = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(bootstrap_estimate_dof)(err_ur, test_size) for _ in range(n_boot)
        )
        dof_ur = np.array(dof_ur)
        dof_r = estimate(err_r)['param'][0]
        p_val.loc[mass, (cols_height[i], cols_height[j])] = np.mean(dof_r < dof_ur)
        pbar.update(1)

# %% Heatmap
fig, ax = plt.subplots(figsize=(0.75 * p_val.shape[1], 0.5 * p_val.shape[0] + 1))
heatmap = sns.heatmap(p_val.values, square=True, linewidths=.5, cmap='coolwarm',
                      vmin=0, vmax=0.1, annot=decimal_non_zero(p_val.values), fmt='', ax=ax)
ax.set_ylabel('Mass')
ax.set_xlabel('(Cause, Effect)')
ax.set_xticklabels(p_val.columns, rotation=45)
ax.set_yticklabels(p_val.index, rotation=0)
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
fig.savefig(f'results/17_{dataset_name}_p_{lag}.eps')
plt.close(fig)

# %% Export.
pd.to_pickle(p_val, f'raw/17_{dataset_name}_p_{lag}.pkl')
