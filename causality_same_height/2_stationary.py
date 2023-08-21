import pickle
import numpy as np
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Constants.
subset_id = 1
height = [80, 150, 320]
p = {}

for h in height:
    # Load data.
    with open(f"raw/1_ptrms_std_{subset_id}_{h}.pkl", "rb") as f:
        dataset = pickle.load(f)
    ptrms_train = dataset.get('training_set')

    # Stationary test. (H0: has unit root = non-stationary)
    ptrms_adfuller = np.apply_along_axis(func1d=adfuller, axis=0, arr=ptrms_train)
    ptrms_adfuller = pd.DataFrame(data=ptrms_adfuller.T,
                                  columns=['adf', 'pvalue', 'usedlag', 'nobs', 'critical values', 'icbest'],
                                  index=ptrms_train.columns)
    p[f'height={h}'] = ptrms_adfuller['pvalue']

p = pd.DataFrame(p)
p.to_csv("results/2_ptrms_height_stationary.csv", index_label='Mass')
