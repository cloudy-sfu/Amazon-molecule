import os
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import wilcoxon
from tqdm import tqdm

# %% Constants.
lag = 48
dataset_name = 'default_15min'

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
os.makedirs(f'raw/3_{dataset_name}_nn_{lag}', exist_ok=True)
w_val = {height: np.full(shape=(len(cols_), len(cols_)), fill_value=np.nan)
         for height, cols_ in cols_grouped_height.items()}
p_val = deepcopy(w_val)
pbar = tqdm(total=sum(len(x)**2 for x in cols_grouped_height.values()))

# %% Infer causal network per height.
for height, cols_this_height in cols_grouped_height.items():
    x_test, y_test = moving_window(ts_test[cols_this_height].values, k=lag)

    for j in range(x_test.shape[2]):
        x_test_ur = x_test
        y_test_ur = y_test[:, j]
        ur = tf.keras.models.load_model(f'raw/3_{dataset_name}_nn_{lag}/{height}_{j}_ur.h5')
        pbar.update(1)

        for i in range(x_test.shape[2]):
            if i == j:
                continue
            x_test_r = x_test.copy()
            rng.shuffle(x_test_r[:, :, i])
            y_test_r = y_test_ur
            r = tf.keras.models.load_model(f'raw/3_{dataset_name}_nn_{lag}/{height}_{i}_{j}_r.h5')
            pbar.update(1)

            # Use model to predict
            y_test_ur_hat = ur.predict(x_test_ur, batch_size=2500, verbose=0)
            y_test_r_hat = r.predict(x_test_r, batch_size=2500, verbose=0)

            # Infer causality
            err_ur = (y_test_ur_hat.flatten() - y_test_ur) ** 2
            err_r = (y_test_r_hat.flatten() - y_test_r) ** 2
            wilcoxon_results = wilcoxon(x=err_r, y=err_ur, method='approx', alternative='greater')
            w_val[height][i, j] = wilcoxon_results.statistic
            p_val[height][i, j] = wilcoxon_results.pvalue

# %% Export.
pd.to_pickle(w_val, f'raw/7_{dataset_name}_w_{lag}.pkl')
pd.to_pickle(p_val, f'raw/7_{dataset_name}_p_{lag}.pkl')
