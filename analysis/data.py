import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

SILSO_DATAPATH = '../data/SILSO data/'
SILSO_DATACOLS = {
    'SN_m_tot_V2.0': ['year', 'month', 'yer_frac', 'ssn_total', 'ssn_stdev', 'nobs', 'marker']
}

def get_silso_data(table):
    data_path = os.path.join(SILSO_DATAPATH, f'{table}.csv')
    print(data_path)
    return pd.read_csv(data_path, sep=';',  names=SILSO_DATACOLS[table])


def centre_x(data):
    data['year_frac'] = data['year'] + data['month'] * 1/12 - 1/24
    return data


def split_train_test(data, min_test_year, min_train_year = 0, year_col='year'):
    traindata = data.loc[(min_train_year <= data[year_col]) & (data[year_col] < min_test_year)]
    testdata = data.loc[data[year_col] >= min_test_year]
    return traindata, testdata

def to_tensor(data, dtype=torch.float32):
    return torch.tensor(data.values, dtype=dtype)

def sample_random(data, n):
    i = np.random.choice(data.index, n)
    i.sort()
    return data.loc[i]

def sample_interval(data, k):
    return data.loc[data.index[::k]]

def sample_meaninterval(data, k, cols):
    data2 = data.copy()
    data2['_group'] = [i // k for i in data.index]
    m = data2.groupby('_group').mean()
    return m.loc[:, cols].reset_index(drop=True)

def plot_data(data, xcol, ycol):
    fig, ax = plt.subplots(figsize=(30, 8))
    ax.scatter(data[xcol], data[ycol])
    return fig, ax