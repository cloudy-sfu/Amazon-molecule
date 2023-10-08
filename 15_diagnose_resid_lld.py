from argparse import ArgumentParser
import pandas as pd
from mle import *
from itertools import product

# %% Constants.
parser = ArgumentParser()
parser.add_argument('--lag', type=int)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--height', type=str, help='for specific height/mass and dependent time series')
parser.add_argument('--mass', type=str, help='for specific height/mass and dependent time series')
parser.add_argument('--mode', type=str, help="mode is 'same_height' or 'same_molecule'")
parser.add_argument('--j', type=int, help='dependent time series in the subset of columns of specific height/mass')
command, _ = parser.parse_known_args()
lag = command.lag
dataset_name = command.dataset_name
height = command.height
mass = command.mass
mode = command.mode
j = command.j

# %% Load data.
# Each row represents results from one model (among 1000 bootstrapped models), each column represents one sample in
# the testing set.
if mode == 'same_height':
    resid = pd.read_pickle(f'raw/14_{dataset_name}_residuals_{lag}_{height}_{j}_ur.pkl')
elif mode == 'same_molecule':
    resid = pd.read_pickle(f'raw/14_{dataset_name}_residuals_{lag}_{mass}_{j}_ur.pkl')
else:
    raise Exception("'mode' should be in ('same_height', 'same_molecule').")

# %% Likelihood ratio based confidence interval.
param_hat = np.zeros(shape=(resid.shape[0], 2))  # point estimate
param_var = np.zeros(shape=(resid.shape[0], 2, 2))
df_ci_ratio = np.zeros(shape=(resid.shape[0], 2))  # confidence interval of parameter 'df'
for i in range(resid.shape[0]):
    mle_result = estimate(resid[i, :])
    param_hat[i, :] = mle_result['param']
    param_var[i] = mle_result['variance']
    df_ci_ratio[i, :] = ci_df_estimate(resid[i, :], mle_result['param'], mle_result['neg_lld'])['CI']
print('Point estimate of (\'df\', \'nc\'):')
print(param_hat)
print('Confidence interval of \'df\':')
print(df_ci_ratio)
# [Result] 'nc' parameters are all 0 -> residuals degenerate to chi2 distribution.

# %% Wald-test based confidence interval.
df_wald_ci = np.zeros(shape=(resid.shape[0], 2, 2))
for i in range(resid.shape[0]):
    df_wald_ci[i, :, 0] = param_hat[i, :] + 1.96 * np.diag(param_var[i])
    df_wald_ci[i, :, 1] = param_hat[i, :] - 1.96 * np.diag(param_var[i])
# [Result] Comparing 'df_ci' and 'df_wald_ci', profile log-likelihood of 'df' is not asymmetric.
#       -> Not recommend to calculate CI of 'df' by Wald test.

# %% Two samples test.
p = np.full(shape=(resid.shape[0], resid.shape[0]), fill_value=np.nan)
for i, j_ in product(range(resid.shape[0]), repeat=2):
    if i == j_:
        continue
    mle_i = estimate(resid[i, :])
    mle_j = estimate(resid[j_, :])
    p[i, j_] = p_df_estimate(resid[i, :], mle_i['neg_lld'], mle_j['param'])
p = np.round(p, 2)
print('p-values of two samples test:')
print(p)
