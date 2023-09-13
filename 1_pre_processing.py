import yaml
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% Constants.
dataset_name = 'default_daily'

# %% Load data.
with open("subset_time_range.yaml", "r") as f:
    subsets = yaml.safe_load(f.read())
with open("sql/resample.sql", "r") as f:
    sql_ptr_ms = f.read()
c = sqlite3.connect('data/ATTO.db')
pivot_longer = pd.read_sql_query(con=c, sql=sql_ptr_ms, params=subsets[dataset_name], index_col='time_index')
c.close()

# %% Build continues time index.
time_index = np.arange(pivot_longer.index.min(), pivot_longer.index.max() + 1)
pivot_wider = pd.DataFrame(index=time_index)

# %% Merge PTR-MS to panel dataset.
first_height = None
for height, ptrms_per_height in pivot_longer.groupby('height'):
    first_height = first_height or height
    ptrms_per_height.drop(columns=['height'], inplace=True)
    pivot_wider = pd.merge(pivot_wider, ptrms_per_height, 'outer', left_index=True, right_index=True, sort=True,
                           suffixes=(None, f'_h{height}'), validate='1:1')
# Fix: `pd.merge` doesn't write suffix when column name isn't duplicate.
pivot_wider.rename(columns={x: x + f'_h{first_height}' for x in pivot_longer.columns}, inplace=True)

# %% Fill missing values.
pivot_wider.fillna(method='ffill', axis=0, inplace=True)

# %% Split training and testing set.
ts_train, ts_test = train_test_split(pivot_wider, train_size=0.8, shuffle=False)

# %% Remove outliers.
normal_upper_bound = ts_train.mean(axis=0) + 3 * ts_train.std(axis=0)
normal_lower_bound = ts_train.mean(axis=0) - 3 * ts_train.std(axis=0)
ts_train.clip(lower=normal_lower_bound, upper=normal_upper_bound, axis=1, inplace=True)
ts_test.clip(lower=normal_lower_bound, upper=normal_upper_bound, axis=1, inplace=True)

# %% Standardization.
std_ = StandardScaler()
ts_train_std = std_.fit_transform(ts_train)
ts_train_std = pd.DataFrame(data=ts_train_std, columns=pivot_wider.columns, index=ts_train.index)
ts_test_std = std_.transform(ts_test)
ts_test_std = pd.DataFrame(data=ts_test_std, columns=pivot_wider.columns, index=ts_test.index)

# %% Export.
pd.to_pickle(std_, f'raw/1_{dataset_name}_standard_scaler.pkl')
pd.to_pickle([ts_train_std, ts_test_std], f'raw/1_{dataset_name}_std.pkl')
