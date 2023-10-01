import os
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import product

# %% Constants.
lag = 96
dataset_name = 'default_15min'

# %% Load data.
ts_train_valid, _ = pd.read_pickle(f'raw/1_{dataset_name}_std.pkl')

# %% Split by mass.
cols = ts_train_valid.columns
mass_heights = map(lambda x: x.split('_') + [x], cols)
cols_grouped_mass = defaultdict(list)
for mass, height, mass_height in mass_heights:
    cols_grouped_mass[mass].append(mass_height)

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
os.makedirs(f'raw/9_{dataset_name}_nn_{lag}', exist_ok=True)
pbar = tqdm(total=(sum(len(x) ** 2 for x in cols_grouped_mass.values()) - len(cols)) * 2)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, start_from_epoch=100)
tf.keras.utils.set_random_seed(5576)  # not fully repeatable, randomness still occur in training

# %% Infer causal network per mass.
for mass, cols_this_height in cols_grouped_mass.items():
    x_train_valid, y_train_valid = moving_window(ts_train_valid[cols_this_height].values, k=lag)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid, y_train_valid, train_size=0.9, random_state=6637)

    # Build model
    l0 = Input(shape=(x_train.shape[1], x_train.shape[2]))
    l1 = LSTM(32)(l0)
    l2 = Dense(32, activation='relu')(l1)
    l3 = Dense(1, activation='linear')(l2)
    basic_model = tf.keras.Model(l0, l3)

    for i, j in product(range(x_train.shape[2]), repeat=2):
        if i == j:
            continue
        ur_path = f'raw/9_{dataset_name}_nn_{lag}/{mass}_{i}_{j}_ur.h5'
        r_path = f'raw/9_{dataset_name}_nn_{lag}/{mass}_{i}_{j}_r.h5'
        if os.path.exists(ur_path) or os.path.exists(r_path):
            continue
        else:
            with open(ur_path, 'w'):  # placeholders
                pass
            with open(r_path, 'w'):
                pass

        # Make dataset
        x_train_ur = x_train
        y_train_ur = y_train[:, j]
        x_train_r = x_train.copy()
        rng.shuffle(x_train_r[:, :, i])
        y_train_r = y_train_ur

        x_valid_ur = x_valid
        y_valid_ur = y_valid[:, j]
        x_valid_r = x_valid.copy()
        rng.shuffle(x_valid_r[:, :, i])
        y_valid_r = y_valid_ur

        # Train model
        ur = tf.keras.models.clone_model(basic_model)
        ur.compile(optimizer='adam', loss='mse')
        sbm_ur = tf.keras.callbacks.ModelCheckpoint(ur_path, save_best_only=True)
        ur.fit(x_train_ur, y_train_ur, validation_data=(x_valid_ur, y_valid_ur), epochs=5000, batch_size=10000,
               callbacks=[stop_early, sbm_ur], verbose=0)
        pbar.update(1)

        r = tf.keras.models.clone_model(basic_model)
        r.compile(optimizer='adam', loss='mse')
        sbm_r = tf.keras.callbacks.ModelCheckpoint(r_path, save_best_only=True)
        r.fit(x_train_r, y_train_r, validation_data=(x_valid_r, y_valid_r), epochs=5000, batch_size=10000,
              callbacks=[stop_early, sbm_r], verbose=0)
        pbar.update(1)
