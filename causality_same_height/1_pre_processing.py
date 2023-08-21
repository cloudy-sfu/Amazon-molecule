import sqlite3
import numpy as np
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %% Constants.
subset_id = 1
height = 320

# %% Load data.
with open("sql/subset_time_range.yaml", "r") as f:
    subsets = yaml.safe_load(f.read())
with open("sql/resample_ptrms_single_height.sql", "r") as f:
    sql_ptrms = f.read()
c = sqlite3.connect('data/ATTO.db')
ptrms = pd.read_sql(con=c, sql=sql_ptrms, params={
    'start_timestamp': subsets['subsets'][subset_id]['start_timestamp'],
    'end_timestamp': subsets['subsets'][subset_id]['end_timestamp'],
    'height': height
}, index_col='qoh_index')
ptrms.drop(columns=['height'], inplace=True)

# %% Build continues time index.
qoh_index = np.arange(ptrms.index.min(), ptrms.index.max() + 1)
qoh_index = pd.DataFrame(index=qoh_index)
ptrms = pd.merge(ptrms, qoh_index, 'outer', left_index=True, right_index=True, sort=True)

# %% Fill missing values.
ptrms.fillna(method='ffill', axis=0, inplace=True)

# %% Split training and testing set.
ptrms_train, ptrms_test = train_test_split(ptrms, train_size=0.8, shuffle=False)

# %% Remove outliers.
normal_upper_bound = ptrms_train.mean(axis=0) + 3 * ptrms_train.std(axis=0)
normal_lower_bound = ptrms_train.mean(axis=0) - 3 * ptrms_train.std(axis=0)
ptrms_train.clip(lower=normal_lower_bound, upper=normal_upper_bound, axis=1, inplace=True)

# %% Standardization.
std_ = StandardScaler()
ptrms_train_std = std_.fit_transform(ptrms_train)
ptrms_train_std = pd.DataFrame(data=ptrms_train_std, columns=ptrms.columns, index=ptrms_train.index)
ptrms_test_std = std_.transform(ptrms_test)
ptrms_test_std = pd.DataFrame(data=ptrms_test_std, columns=ptrms.columns, index=ptrms_test.index)

# %% Export.
pd.to_pickle(
    {
        'subset_id': subset_id,
        'height': height,
        'standard_scaler': std_,
        'training_set': ptrms_train_std,
        'testing_set': ptrms_test_std,
    },
    f"raw/1_ptrms_std_{subset_id}_{height}.pkl"
)
