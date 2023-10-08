import os
import pandas as pd
from collections import defaultdict
import numpy as np
import tensorflow as tf
from keras.layers import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# %% Constants.
lag = 96
dataset_name = 'default_15min'
n_boot = 500

# %% Load data.
ts_train_valid, ts_test = pd.read_pickle(f'raw/1_{dataset_name}_std.pkl')

# %% Split by mass.
cols = ts_train_valid.columns
mass_heights = map(lambda x: x.split('_') + [x], cols)
cols_grouped_height = defaultdict(list)
for mass, height, mass_height in mass_heights:
    cols_grouped_height[mass].append(mass_height)

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
os.makedirs(f'raw/11_{dataset_name}_nn_{lag}', exist_ok=True)
pbar = tqdm(total=sum(len(x) for x in cols_grouped_height.values()), desc='Overall')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, start_from_epoch=100)
# strategy = tf.distribute.MirroredStrategy()

def check_progress(fm, fj):
    incomplete = []
    for fb in range(n_boot):
        path = f'raw/11_{dataset_name}_nn_{lag}/{fm}_{fj}_{fb}_ur.h5'
        if not os.path.exists(path):
            incomplete.append(path)
    return incomplete

# %% Infer causal network per mass.
for mass, cols_this_height in cols_grouped_height.items():
    x_train_valid, y_train_valid = moving_window(ts_train_valid[cols_this_height].values, k=lag)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid, y_train_valid, train_size=0.9, random_state=6637)
    x_test, y_test = moving_window(ts_test[cols_this_height].values, k=lag)

    # Build model
    # with strategy.scope():
    l0 = Input(shape=(x_train.shape[1], x_train.shape[2]))
    l1 = LSTM(32)(l0)
    l2 = Dense(32, activation='relu')(l1)
    l3 = Dense(1, activation='linear')(l2)
    ur = tf.keras.Model(l0, l3)
    ur.compile(optimizer='adam', loss='mse')
    # end strategy.scope()
    init_weights = ur.get_weights()

    for j in range(x_train.shape[2]):
        ur_paths = check_progress(mass, j)
        for ur_path in tqdm(ur_paths, desc=f'{mass}_{j}'):
            if os.path.exists(ur_path):
                continue
            else:
                with open(ur_path, 'w'):
                    pass
            ur.set_weights(init_weights)
            sbm_ur = tf.keras.callbacks.ModelCheckpoint(ur_path, save_best_only=True)
            ur.fit(x_train, y_train[:, j], validation_data=(x_valid, y_valid[:, j]), epochs=5000, batch_size=10000,
                   callbacks=[stop_early, sbm_ur], verbose=0)
        pbar.update(1)
