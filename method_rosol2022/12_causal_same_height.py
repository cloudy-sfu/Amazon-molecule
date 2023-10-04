import os
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from itertools import product
from method_rosol2022.nonlincausality import run_nonlincausality, LSTM_architecture
from copy import deepcopy

# %% Constants.
lag = 96
dataset_name = 'default_15min'

# %% Load data.
ts_train, ts_test = pd.read_pickle(f'raw/1_{dataset_name}_std.pkl')

# %% Split by height.
cols = ts_train.columns
mass_heights = map(lambda x: x.split('_') + [x], cols)
cols_grouped_height = defaultdict(list)
for mass, height, mass_height in mass_heights:
    cols_grouped_height[height].append(mass_height)

# %% Initialization.
rng = np.random.RandomState(6611)
os.makedirs(f'raw/12_{dataset_name}_nn_{lag}', exist_ok=True)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, start_from_epoch=100)
if os.path.exists(f'raw/12_{dataset_name}_w_{lag}.pkl') and os.path.exists(f'raw/12_{dataset_name}_p_{lag}.pkl'):
    w_val = pd.read_pickle(f'raw/12_{dataset_name}_w_{lag}.pkl')
    p_val = pd.read_pickle(f'raw/12_{dataset_name}_p_{lag}.pkl')
else:
    w_val = {height: np.full(shape=(len(x), len(x)), fill_value=np.nan)
             for height, x in cols_grouped_height.items()}
    p_val = deepcopy(w_val)
pbar = tqdm(total=sum(len(x)**2 for x in cols_grouped_height.values()) - len(cols))
tf.keras.utils.set_random_seed(5576)  # not fully repeatable, randomness still occur in training

# %% Infer causality per height.
for height, cols_this_height in cols_grouped_height.items():
    ts_train_1 = ts_train[cols_this_height].values
    ts_test_1 = ts_test[cols_this_height].values

    for i, j in product(range(ts_train.shape[1]), repeat=2):
        if i == j:
            continue
        ur_path = f'raw/12_{dataset_name}_nn_{lag}/{height}_{i}_{j}_ur'  # only placeholder, cannot save models
        r_path = f'raw/12_{dataset_name}_nn_{lag}/{height}_{i}_{j}_r'
        if os.path.exists(ur_path) or os.path.exists(r_path):
            continue
        else:
            with open(ur_path, 'w'):  # placeholders
                pass
            with open(r_path, 'w'):
                pass

        # column 1 is X, column 2 is Y, package detects Y->X.
        xy_train = ts_train_1[:, [j, i]]
        z_train = np.delete(ts_train_1, [i, j], axis=1)
        xy_test = ts_test_1[:, [j, i]]
        z_test = np.delete(ts_test_1, [i, j], axis=1)

        results = run_nonlincausality(
            network_architecture=LSTM_architecture,
            x=xy_train, x_test=xy_test, z=z_train, z_test=z_test,
            maxlag=[lag], run=1, epochs_num=5000, batch_size_num=xy_train.shape[0], verbose=False, plot=False,
            Network_layers=1, Network_neurons=[32], Dense_layers=1, Dense_neurons=[32], add_Dropout=False,
            function_type='LSTM', regularization=None, reg_alpha=None, callbacks=[stop_early],
            learning_rate=0.001,  # Adam default
            Dropout_rate=0.0,
        )

        w_val[height][i, j] = results[lag].test_statistic
        p_val[height][i, j] = results[lag].p_value

        pd.to_pickle(w_val, f'raw/12_{dataset_name}_w_{lag}.pkl')
        pd.to_pickle(p_val, f'raw/12_{dataset_name}_p_{lag}.pkl')
