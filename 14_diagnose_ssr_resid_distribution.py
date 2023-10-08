import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import contextlib
import seaborn as sns
from argparse import ArgumentParser
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

# %% Constants.
parser = ArgumentParser()
parser.add_argument('--lag', type=int)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--n_boot_height', type=int, help='number of instances of \'same_height\'', default=0)
parser.add_argument('--n_boot_molecule', type=int, help='number of instances of \'same_molecule\'', default=0)
parser.add_argument('--height', type=str, help='for specific height/mass and dependent time series')
parser.add_argument('--mass', type=str, help='for specific height/mass and dependent time series')
parser.add_argument('--mode', type=str, help="mode is 'same_height' or 'same_molecule'")
parser.add_argument('--j', type=int, help='dependent time series in the subset of columns of specific height/mass')
command, _ = parser.parse_known_args()
lag = command.lag
dataset_name = command.dataset_name
n_boot_height = command.n_boot_height
n_boot_molecule = command.n_boot_molecule
height_ = command.height
mass_ = command.mass
mode = command.mode
j = command.j

# %% Load data.
_, ts_test = pd.read_pickle(f'raw/1_{dataset_name}_std.pkl')

# %% Split by height.
cols = ts_test.columns
mass_heights = map(lambda x: x.split('_') + [x], cols)
cols_grouped_height = defaultdict(list)
cols_grouped_mass = defaultdict(list)
for mass, height, mass_height in mass_heights:
    cols_grouped_height[height].append(mass_height)
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

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    Reference: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# %% Estimate MSE of all bootstrapping UR models.
def predict_1(model_path, x, y):
    try:
        model = tf.keras.models.load_model(model_path)
    except (OSError, TypeError):
        return np.nan
    y_hat = model.predict(x, batch_size=x.shape[0], verbose=0)
    del model
    mse = np.mean((y_hat.flatten() - y) ** 2)
    return mse

if mode == 'same_height':
    cols_this_mass = cols_grouped_height[height_]
    x_test, y_test = moving_window(ts_test[cols_this_mass].values, k=lag)
    x_test_ur = x_test
    y_test_ur = y_test[:, j]
    ur_paths = [f'raw/5_{dataset_name}_nn_{lag}/{height_}_{j}_{b}_ur.h5' for b in range(n_boot_height)]

    with tqdm_joblib(tqdm_object=tqdm(total=n_boot_height, desc=f'{height_}_{j}')):
        mse_ur = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(predict_1)(ur_path, x_test_ur, y_test_ur) for ur_path in ur_paths
        )
elif mode == 'same_molecule':
    cols_this_mass = cols_grouped_mass[mass_]
    x_test, y_test = moving_window(ts_test[cols_this_mass].values, k=lag)
    x_test_ur = x_test
    y_test_ur = y_test[:, j]
    ur_paths = [f'raw/11_{dataset_name}_nn_{lag}/{mass_}_{j}_{b}_ur.h5' for b in range(n_boot_molecule)]

    with tqdm_joblib(tqdm_object=tqdm(total=n_boot_molecule, desc=f'{mass_}_{j}')):
        mse_ur = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(predict_1)(ur_path, x_test_ur, y_test_ur) for ur_path in ur_paths
        )
else:
    raise Exception("'mode' should be in ('same_height', 'same_molecule').")
mse_ur = np.array(mse_ur)
mse_nan = np.isnan(mse_ur)
mse_nan_n = np.sum(mse_nan).astype(int)
if mse_nan_n > 0:
    mse_ur = mse_ur[~mse_nan]

# %% All MSE histogram.
fig, ax1 = plt.subplots(figsize=(7.5, 6))
if mode == 'same_height':
    hist_range = [-0.05, 0.5]
elif mode == 'same_molecule':
    hist_range = [0, 0.1]
ax1.hist(mse_ur, bins=21, range=hist_range, edgecolor='k', color='#808080')
ax1.set_xlabel('MSE')
ax1.set_ylabel('Count')
ax1.set_xlim(hist_range)
ax2 = ax1.twinx()
sns.kdeplot(mse_ur, ax=ax2, color='k')
ax2.set_ylabel('Probability Density')
if mode == 'same_height':
    fig.savefig(f'results/14_{dataset_name}_all_MSE_{lag}_{height_}_{j}.eps')
elif mode == 'same_molecule':
    fig.savefig(f'results/14_{dataset_name}_all_MSE_{lag}_{mass_}_{j}.eps')
plt.close(fig)

# %% Residuals.
fig, ax = plt.subplots(figsize=(7.5, 6))
drawn_ur_paths = rng.choice(ur_paths, 20, replace=False)
residuals = []
for ur_path in tqdm(drawn_ur_paths):
    try:
        model = tf.keras.models.load_model(ur_path)
    except (OSError, TypeError):
        continue
    y_hat = model.predict(x_test_ur, batch_size=x_test_ur.shape[0], verbose=0)
    resid = (y_hat.flatten() - y_test_ur) ** 2
    residuals.append(resid)
    sns.kdeplot(resid, ax=ax, label=os.path.splitext(os.path.basename(ur_path))[0])
ax.set_xlabel('Residual')
ax.set_ylabel('Probability Density')
ax.set_xlim([-0.05, 0.5])
plt.legend()
if mode == 'same_height':
    fig.savefig(f'results/14_{dataset_name}_residuals_{lag}_{height_}_{j}.eps')
elif mode == 'same_molecule':
    fig.savefig(f'results/14_{dataset_name}_residuals_{lag}_{mass_}_{j}.eps')
plt.close(fig)

# %% Export residuals.
residuals = np.array(residuals)
if mode == 'same_height':
    pd.to_pickle(residuals, f'raw/14_{dataset_name}_residuals_{lag}_{height_}_{j}_ur.pkl')
elif mode == 'same_molecule':
    pd.to_pickle(residuals, f'raw/14_{dataset_name}_residuals_{lag}_{mass_}_{j}_ur.pkl')
