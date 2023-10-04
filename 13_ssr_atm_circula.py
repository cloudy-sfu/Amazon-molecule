import pandas as pd
import numpy as np

# %% Constant.
lag = 96
dataset_name = 'default_15min'

# %% Load data.
p_height = pd.read_pickle(f'raw/6_{dataset_name}_p_{lag}.pkl')
p_mass = pd.read_pickle(f'raw/12_{dataset_name}_p_{lag}.pkl')

# %% Transform data format.
causality = {}
for height, pmat in p_height.items():
    for i, j in zip(*np.where(pmat < 0.05)):
        cause, effect = p_mass.index[i], p_mass.index[j]
        if (cause, effect) in causality.keys():
            continue
        causality[(cause, effect)] = {'cause': cause, 'effect': effect}
        for height_1, pmat_1 in p_height.items():
            causality[(cause, effect)][height_1] = pmat_1[i, j]
        for atm_pair in p_mass.columns:
            causality[(cause, effect)][('cause', ) + atm_pair] = p_mass.loc[cause, atm_pair]
            causality[(cause, effect)][('effect', ) + atm_pair] = p_mass.loc[effect, atm_pair]
causality = list(causality.values())
causality = pd.DataFrame(causality)

# %% Export.
causality.to_excel(f'results/13_{dataset_name}_p_{lag}.xlsx', index=False)
