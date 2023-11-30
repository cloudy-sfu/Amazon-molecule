import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tqdm import tqdm
import joblib
import contextlib
from datetime import datetime
from itertools import product

# %% Constants.
lag = 96
dataset_name = 'default_15min'
n_boot = 400

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
os.makedirs(f'results/12_{dataset_name}_p/', exist_ok=True)
p_val = pd.DataFrame(index=cols_grouped_mass.keys(), columns=pd.MultiIndex.from_tuples(heights_pair), dtype=float)
pbar = tqdm(total=sum(len(x) ** 2 for x in cols_grouped_mass.values()) - len(cols), desc='Overall')
log_path = f'raw/12_{dataset_name}_log_{lag}_{datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")}'

@np.vectorize
def decimal_non_zero(x):
    return format(x, '.2f').removeprefix('0')

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

# %% Infer causal network per height.
def predict_1(model_path, x, y):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        with open(log_path, 'a') as g:
            g.write(f'model_path={model_path}, error={e}.\n')
        return np.nan
    y_hat = model.predict(x, batch_size=x.shape[0], verbose=0)
    del model
    mse = np.mean((y_hat.flatten() - y) ** 2)
    return mse

for mass, cols_this_mass in cols_grouped_mass.items():
    x_test, y_test = moving_window(ts_test[cols_this_mass].values, k=lag)
    n_series = x_test.shape[2]
    cols_height = [col.split('_')[1] for col in cols_this_mass]

    for j in range(n_series):
        x_test_ur = x_test
        y_test_ur = y_test[:, j]
        ur_paths = [f'raw/11_{dataset_name}_nn_{lag}/{mass}_{j}_{b}_ur.h5' for b in range(n_boot)]
        with tqdm_joblib(tqdm_object=tqdm(total=n_boot, desc=f'{mass}_{j}')):
            mse_ur = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(predict_1)(ur_path, x_test_ur, y_test_ur) for ur_path in ur_paths
            )
        mse_ur = np.array(mse_ur)
        mse_nan = np.isnan(mse_ur)
        mse_nan_n = np.sum(mse_nan).astype(int)
        if mse_nan_n > 0:
            with open(log_path, 'a') as f:
                f.write(f'height={mass}, dependent={j}, #nan={mse_nan_n}.\n')
            mse_ur = mse_ur[~mse_nan]

        for i in range(n_series):
            if i == j:
                continue
            x_test_r = x_test.copy()
            rng.shuffle(x_test_r[:, :, i])
            y_test_r = y_test_ur
            mse_r = predict_1(f'raw/9_{dataset_name}_nn_{lag}/{mass}_{i}_{j}_r.h5', x_test_r, y_test_r)
            if np.isnan(mse_r):
                with open(log_path, 'a') as f:
                    f.write(f'height={mass}, dependent={j}, independent={i}, error=R model doesn\'t exist.\n')
                p_val.loc[mass, (cols_height[i], cols_height[j])] = np.nan
            else:
                p_val.loc[mass, (cols_height[i], cols_height[j])] = np.mean(mse_r < mse_ur)  # H_0: SSR_r <= SSR_ur
            pbar.update(1)

# %% Heatmap.
fig, ax = plt.subplots(figsize=(9, 4))
heatmap = sns.heatmap(p_val.values.T, square=True, linewidths=.5, cmap='coolwarm', vmin=0, vmax=0.1,
                      annot=decimal_non_zero(p_val.values.T), fmt='', ax=ax)
ax.set_xlabel('m/Q')
ax.set_ylabel('(Cause, Effect)')
ax.set_yticklabels(p_val.columns, rotation=0)
ax.set_xticklabels(p_val.index, rotation=0)
fig.subplots_adjust(left=0.16, right=1.05, bottom=0.05, top=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
fig.savefig(f'results/12_{dataset_name}_p_{lag}.eps')
plt.close(fig)

# %% Export.
pd.to_pickle(p_val, f'raw/12_{dataset_name}_p_{lag}.pkl')
