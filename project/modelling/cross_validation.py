import pandas as pd
import xarray as xr
import xgboost as xgb
import numpy as np
from itertools import product
from pathlib import Path
import os

import multiprocessing

load_path = Path(os.getcwd()).parents[0]/Path('dataset/datasets/model_ready/train.csv')
train_data = pd.read_csv(load_path, index_col='Label')

X = train_data[['max_ocean_SLP_gradient', 'max_landfalling_v850hPa', 'max_landfalling_omega500', 'max_IWV_ais', 'cumulative_landfalling_area', 'max_south_extent']]
snow_var = 'cumulative_snowfall_ais'
temp_var = 'max_2m_temp_ais'


# some basic cross validation
# for snowfall
etas = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3])
# units of Y are in gigatons, a loss of 1 will be really
# high, so that seems like a reasonable upper bound
gammas = np.array([0.0005, 0.001, 0.005, 0.01, 0.05])
# intuition tells me we won't need to go THAT deep
max_depth = np.array([3, 7, 10, 15, 30])
# don't have a great intuition on this, mostly testing
lambdas = np.array([0, 0.3, 0.7, 1, 5])
# same
min_child_weights = np.array([0.0005, 0.001, 0.005, 0.01, 0.05])
# since we're doing early stopping, this should be good
nrounds = 500
early_stopping_rounds = 10
tree_method = 'exact'
booster = 'gbtree'

hyperparams_lst = list(product(gammas, max_depth, lambdas, min_child_weights))

# function that will be run in parallel on chunks on the hyperparameters
def process_hyperparam_chunk(lst):
    num_eta = len(etas)
    results_lst = np.zeros((len(etas)*len(lst), len(lst[0]) + 3))
    
    for i, param_set in enumerate(lst):
        for j, eta in enumerate(etas):
            params = dict(booster='gbtree',
                   eta=eta,
                   gamma=param_set[0],
                   max_depth=param_set[1],
                   reg_lambda=param_set[2],
                   min_child_weight=param_set[3],
                   tree_method=tree_method)
            dtrain = xgb.DMatrix(X, label=train_data[snow_var])
            res = xgb.cv(params,
                   dtrain,
                   nrounds,
                   nfold=5,
                   seed=12345,
                   early_stopping_rounds=early_stopping_rounds)
            results_lst[i*num_eta + j,:] = np.array(list(param_set) + [eta] + [res.shape[0]] + [float(res['test-rmse-mean'].iloc[-1])])

            results_df = pd.DataFrame(res, columns=['gamma', 'max_depth', 'reg_lambda', 'min_child_weight', 'eta', 'num_boost', 'test_rmse'])

    return results_df

# creating the chunks of hyperparameters
chunk_lst = []
chunk_size = 100


for i in range(0, len(hyperparams_lst), chunk_size):
    chunk_lst.append(hyperparams_lst[i:i + chunk_size])

n_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

if __name__ == '__main__':
    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.map(process_hyperparam_chunk, chunk_lst)
    full_df = pd.concat(results)
    full_df.to_csv('cv_res.csv')
