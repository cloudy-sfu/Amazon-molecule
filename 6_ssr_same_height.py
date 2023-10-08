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

# %% Constants.
lag = 96
dataset_name = 'default_15min'
n_boot = 1000

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
os.makedirs(f'results/6_{dataset_name}_p/', exist_ok=True)
p_val = {height: np.full(shape=(len(x), len(x)), fill_value=np.nan)
         for height, x in cols_grouped_height.items()}
pbar = tqdm(total=sum(len(x)**2 for x in cols_grouped_height.values()) - len(cols), desc='Overall')
log_path = f'raw/6_{dataset_name}_log_{lag}_{datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")}'

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

for height, cols_this_height in cols_grouped_height.items():
    x_test, y_test = moving_window(ts_test[cols_this_height].values, k=lag)
    n_series = x_test.shape[2]

    for j in range(n_series):
        x_test_ur = x_test
        y_test_ur = y_test[:, j]
        ur_paths = [f'raw/5_{dataset_name}_nn_{lag}/{height}_{j}_{b}_ur.h5' for b in range(n_boot)]
        with tqdm_joblib(tqdm_object=tqdm(total=n_boot, desc=f'{height}_{j}')):
            mse_ur = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(predict_1)(ur_path, x_test_ur, y_test_ur) for ur_path in ur_paths
            )
        mse_ur = np.array(mse_ur)
        mse_nan = np.isnan(mse_ur)
        mse_nan_n = np.sum(mse_nan).astype(int)
        if mse_nan_n > 0:
            with open(log_path, 'a') as f:
                f.write(f'height={height}, dependent={j}, #nan={mse_nan_n}.\n')
            mse_ur = mse_ur[~mse_nan]

        for i in range(n_series):
            if i == j:
                continue
            x_test_r = x_test.copy()
            rng.shuffle(x_test_r[:, :, i])
            y_test_r = y_test_ur
            mse_r = predict_1(f'raw/3_{dataset_name}_nn_{lag}/{height}_{i}_{j}_r.h5', x_test_r, y_test_r)
            if np.isnan(mse_r):
                with open(log_path, 'a') as f:
                    f.write(f'height={height}, dependent={j}, independent={i}, error=R model doesn\'t exist.\n')
                p_val[height][i, j] = np.nan
            else:
                p_val[height][i, j] = np.mean(mse_r < mse_ur)  # H_0: SSR_r > SSR_ur
            pbar.update(1)

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
    fig.savefig(f'results/6_{dataset_name}_p/{lag}_{height}.eps')
    plt.close(fig)

pd.to_pickle(p_val, f'raw/6_{dataset_name}_p_{lag}.pkl')
