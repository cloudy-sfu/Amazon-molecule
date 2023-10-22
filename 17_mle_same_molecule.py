from collections import defaultdict
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
from itertools import product
from mle import *
import joblib
import logging
from scipy.stats import chi2

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
rng = np.random.RandomState(3202)
p_val = pd.DataFrame(index=cols_grouped_mass.keys(), columns=pd.MultiIndex.from_tuples(heights_pair), dtype=float)
pbar = tqdm(total=sum(len(x) ** 2 for x in cols_grouped_mass.values()) - len(cols))
test_size = ts_test.shape[0] - lag
test_size_range = np.arange(test_size)
boot_seed_father = rng.randint(low=n_boot, high=n_boot * 1000, size=1)
boot_seed_seq = np.arange(boot_seed_father, boot_seed_father + n_boot)
logging.basicConfig(filename=f'raw/17_{dataset_name}_log_{lag}.txt', filemode='w', level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

@np.vectorize
def decimal_non_zero(x):
    return format(x, '.2f').removeprefix('0')

def bootstrap_estimate_f(err_ur_all, err_r_all, n, seed):
    rng_boot = np.random.RandomState(boot_seed_seq[seed])
    idx = rng_boot.choice(test_size_range, size=n, replace=True)
    err_ur_boot = err_ur_all[idx]
    err_r_boot = err_r_all[idx]
    f_boot = np.log(np.var(err_ur_boot, ddof=1)) - np.log(np.var(err_r_boot, ddof=1))
    return f_boot

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
        err_ur = (y_test_ur_hat.flatten() - y_test_ur) ** 2
        err_r = (y_test_r_hat.flatten() - y_test_r) ** 2

        # Infer causality
        f_hat = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(bootstrap_estimate_f)(err_ur, err_r, test_size, k) for k in range(n_boot)
        )
        f_hat = np.maximum(f_hat, 0)
        mle_result = estimate(ts_test.shape[0] * f_hat)
        m, L_f = mle_result['param']
        if np.isnan(mle_result['neg_lld']):
            logging.warning(f'mass={mass}, {i}->{j} is a bad MLE fit.')
        p = 1 - chi2.cdf(L_f, df=m)
        p_val.loc[mass, (cols_height[i], cols_height[j])] = p
        logging.info(f"mass={mass}, {i}->{j}, L*F={L_f} (mean), m={m} (df), p={p}, LLD={-mle_result['neg_lld']}.")
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
