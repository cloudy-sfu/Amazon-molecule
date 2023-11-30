import pandas as pd
from statsmodels.tsa.stattools import adfuller, pacf, acf
import numpy as np
import matplotlib.pyplot as plt

# %% Load data.
default_15min_train, default_15min_test = pd.read_pickle(f'raw/1_default_15min_std.pkl')

# %% Stationary: test whether the time series are stationary.
#     H_0: Time series contains a unit root and is non-stationary.
#     H_1: Time series doesn't contain a unit root and is stationary.
adf = np.apply_along_axis(func1d=adfuller, axis=0, arr=default_15min_train)
adf = pd.DataFrame(data=adf, index=['adf', 'pvalue', 'usedlag', 'nobs', 'critical values', 'icbest'],
                   columns=default_15min_train.columns)
adf.to_excel(f'results/2_default_15min_stationary.xlsx')
# [Result] Confidence level = 0.05, all time series are stationary.

# %% PACF test: determine `p` in ARIMA model.
# https://en.wikipedia.org/wiki/Partial_autocorrelation_function
pacf_ = np.apply_along_axis(func1d=lambda x: pacf(x, nlags=30), axis=0, arr=default_15min_train)
pacf_ = pd.DataFrame(data=pacf_, columns=default_15min_train.columns)
pacf_.to_excel(f'results/2_default_15min_pacf.xlsx', index_label='Lag')
# [Result] Confidence level = 0.05, p = 7

# %% ACF test: determine `q` in ARIMA model.
# https://otexts.com/fpp2/non-seasonal-arima.html (also: auto correlation)
# Method to determine max lag: enlarge when all lags are significant.
acf_ = np.apply_along_axis(func1d=lambda x: acf(x, nlags=500), axis=0, arr=default_15min_train)
acf_ = pd.DataFrame(data=acf_, columns=default_15min_train.columns)
acf_.to_excel(f'results/2_default_15min_acf.xlsx', index_label='Lag')
# [Result] Confidence level = 0.05, q is infinity or very large.
# Dr. Jonny suggest q > 100 is unrealistic to fit ARIMA.

# %% ACF line plot.
heights, heights_count = np.unique([x.split('_h')[1] for x in acf_.columns], return_counts=True)
n_heights = heights.shape[0]
fig, axes = plt.subplots(nrows=n_heights, figsize=(9, 12))
heights_count_cumsum = np.cumsum(heights_count)
for i in range(n_heights):
    ax = axes[i]
    # US English: gray; British English: grey
    colors = plt.cm.Greys(np.linspace(0.2, 1, heights_count[i]))
    acf_col_names = acf_.columns[heights_count_cumsum[i] - heights_count[i]:heights_count_cumsum[i]]
    for c, col in zip(colors, acf_col_names):
        ax.plot(acf_[col], color=c, label=col)
    ax.set_xlabel('Lag Item')
    ax.set_ylabel('p')
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
fig.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(f'results/2_default_15min_acf_line.eps')
plt.close(fig)
# [Result] Combine figure and data, there's very regular spike every 96 time steps.
#       -> Resample dataset every 1 day (15min * 96).

# %% Load data.
default_daily_train, default_daily_test = pd.read_pickle(f'raw/1_default_daily_std.pkl')

# %% Stationary.
adf_1 = np.apply_along_axis(func1d=adfuller, axis=0, arr=default_daily_train)
adf_1 = pd.DataFrame(data=adf_1, index=['adf', 'pvalue', 'usedlag', 'nobs', 'critical values', 'icbest'],
                     columns=default_daily_train.columns)
adf_1.to_excel(f'results/2_default_daily_stationary.xlsx')
# [Result] Not stationary -> 1st differentiate

# %% Stationary (diff=1).
default_daily_train_1 = np.diff(default_daily_train, n=1, axis=0)
adf_2 = np.apply_along_axis(func1d=adfuller, axis=0, arr=default_daily_train_1)
adf_2 = pd.DataFrame(data=adf_2, index=['adf', 'pvalue', 'usedlag', 'nobs', 'critical values', 'icbest'],
                     columns=default_daily_train.columns)
adf_2.to_excel(f'results/2_default_daily_stationary_diff_1.xlsx')
# [Result] Stationary.

# %% ACF test -> q
acf_1 = np.apply_along_axis(func1d=lambda x: acf(x, nlags=100), axis=0, arr=default_daily_train_1)
acf_1 = pd.DataFrame(data=acf_1, columns=default_daily_train.columns)
acf_1.to_excel(f'results/2_default_daily_acf.xlsx', index_label='Lag')
# [Result] No obvious rule, let q = 0.

# %% PACF test -> p
pacf_1 = np.apply_along_axis(func1d=lambda x: pacf(x, nlags=30), axis=0, arr=default_daily_train_1)
pacf_1 = pd.DataFrame(data=pacf_1, columns=default_daily_train.columns)
pacf_1.to_excel(f'results/2_default_daily_pacf.xlsx', index_label='Lag')
# [Result] No obvious rule， let p = 0.
#       -> Since p, q = 0, there is no periodicity.

# %% FFT (diff=0).
n = default_daily_train.shape[0]  # sample size of training set

def fft_p_value(ts):
    np.fft.rfft(ts)
    fft_values = np.fft.rfft(ts)
    fft_abs = np.abs(fft_values)
    return fft_abs

fft = np.apply_along_axis(func1d=fft_p_value, axis=0, arr=default_daily_train)
period_length = np.round(1 / np.fft.rfftfreq(n))

def array_group_by(a: np.ndarray):
    """
    Group array by unique values.
    https://stackoverflow.com/questions/23159791/find-the-indices-of-non-zero-elements-and-group-by-values
    :param a: 1-D numeric numpy array
    :return: List[np.ndarray, ...] each ndarray is a group of indices of one unique value.
    """
    sorted_idx = np.argsort(a)
    sorted_a = a[sorted_idx]
    a_sorted_changes = np.where(np.diff(sorted_a))[0] + 1
    return np.split(sorted_idx, a_sorted_changes)

period_value = np.sort(np.unique(period_length))
period_indices = array_group_by(period_length)
fft_1 = np.full(shape=(period_value.shape[0], fft.shape[1]), fill_value=np.nan)
for i in range(period_value.shape[0]):
    fft_1[i, :] = np.max(fft[period_indices[i], :], axis=0)
fft_1 = pd.DataFrame(data=fft_1, columns=default_daily_train.columns, index=period_value)
fft_1.to_excel(f'results/2_default_daily_fft.xlsx', index_label='Period_length')
# [Result] No obvious rule.
#       -> Fallback to ARIMA and poly root.
