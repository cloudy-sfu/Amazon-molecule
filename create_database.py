"""
Convert dataset from BVOCsATTO_betaVersion.csv to sqlite3 database.
"""
import sqlite3
import pandas as pd
import numpy as np

# %% Load the data.
emission = pd.read_csv("pre_processing/BVOCsATTO_betaVersion.csv")
c = sqlite3.connect("pre_processing/ATTO.db")

# %% Time.
emission['timestamp'] = np.round(emission['Unnamed: 0'], decimals=0).astype(int)

# %% Height.
emission['height'] = emission['height'].apply(lambda x: int(x.removesuffix('m')))

# %% Measurement (ppb).
mass_cols = ['33.033491', '42.033826', '45.033491', '59.049141', '63.026298',
             '69.069877', '71.049141', '73.064791', '79.054227', '93.069877',
             '107.085527', '121.101177', '137.132477', '180.93732', '205.195077',
             '371.101233']
ptrms_cols = ['timestamp', 'height'] + mass_cols
ptrms = emission[ptrms_cols].copy()
ptrms.rename(columns={mass_cols[i]: f'M{i + 1}' for i in range(len(mass_cols))}, inplace=True)
ptrms.to_sql(name="PTRMS", con=c, if_exists='replace', index=False)

# %% Mass.
mass_dict = [
    {'col_name': f'M{i + 1}', 'mass': float(mass_cols[i])}
    for i in range(len(mass_cols))
]
mass_dict = pd.DataFrame(mass_dict)
mass_dict.to_sql(name="mass", con=c, if_exists='replace', index=False)

# %% LOD profile 3.5min.
lod_profile_cols = []
lod_profile = []
n = emission.index.shape[0]
for col in emission.columns:
    if col.startswith("LOD (profile 3.5min)"):
        lod_profile_cols.append(col)
        mass = float(col.split(': ')[1])
        emission_this_col = emission[col].fillna(value=np.inf)
        diff = emission_this_col.diff()
        runs_indices = np.where(np.abs(diff) > 1e-10)[0]
        runs_indices = np.hstack((0, runs_indices, n))
        for i in range(runs_indices.shape[0] - 1):
            index_from = emission.index[runs_indices[i]]
            index_to = emission.index[runs_indices[i+1] - 1]
            lod_profile.append({
                'mass': mass,
                'timestamp_from': emission['timestamp'][index_from],
                'timestamp_to': emission['timestamp'][index_to],
                'value': emission[col][index_from],
            })
lod_profile = pd.DataFrame(lod_profile)
lod_profile.to_sql(name="LOD_profile", con=c, if_exists='replace', index=False)

# %% Close database.
c.close()
