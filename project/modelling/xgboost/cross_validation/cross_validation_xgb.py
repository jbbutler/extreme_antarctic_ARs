'''
Main script that performs the hyperparameter search using CV, for the xgboost models.
Meant to be called from slurm scripts.

Jimmy Butler
October 2025
'''

import pandas as pd
import xarray as xr
import xgboost as xgb
import numpy as np
from itertools import product
from pathlib import Path
from tqdm import tqdm
import os
import json
import argparse
from cv_utils import *
from functools import partial
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import multiprocessing
from cv_constants import XGB_SEED, FOLD_SEED

# parsing CV args to script
parser = argparse.ArgumentParser(description='Parser for CV arguments in hyperparameter search.')
parser.add_argument('--x_cols', nargs='+', required=True, help='A list of feature columns to use.')
parser.add_argument('--y_col', type=str, required=True, help='A string indicating target y variable.')
parser.add_argument('--hyperparam_json', type=str, required=True, help='File path to json with this round\'s CV hyperparameters.')
parser.add_argument('--chunk_size', type=int, default=100, help='How many hyperparam sets each worker will process in parallel.')
parser.add_argument('--save_name', type=str, required=True, help='Name of file with CV round results.')
parser.add_argument('--ncores', type=int, help='How many cores to use in the parallelization.')
parser.add_argument('--shrink', action='store_true', help='Should we use predictive shrinkage?')

load_path = Path(os.getcwd()).parents[2]/Path('dataset/datasets/model_ready/train.csv')

args = parser.parse_args()

with open(os.getcwd() + '/rounds/' + args.hyperparam_json, 'r') as file:
    hyperparam_dict = json.load(file)
hyperparams_lst = [lst for key, lst in hyperparam_dict.items() if key in ['gammas', 'max_depth', 'lambdas', 'min_child_weights', 'subsample_fracs']]
hyperparams_lst = list(product(*hyperparams_lst))

# creating the chunks of hyperparameters
chunk_lst = []
chunk_size = args.chunk_size

for i in range(0, len(hyperparams_lst), chunk_size):
    chunk_lst.append(hyperparams_lst[i:i + chunk_size])
    
ncores = args.ncores
if ncores == None:
    ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

# load data
train_data = pd.read_csv(load_path, index_col='Label')
X = train_data[args.x_cols]
y = train_data[args.y_col]

# prepare training and validation sets for k-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)

fit_ols = (args.y_col == 'cumulative_snowfall_ais')

# list of lists: each list stores training, validation, and any extra models if need be
data_CV_splits = []
for train_idx, val_idx in kf.split(X, y):

    fold_dict = {}
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    y_train_centered = y_train - y_train.mean()
    y_val_centered = y_val - y_val.mean()

    fold_dict['y_val_centered'] = y_val_centered
    fold_dict['X_train'] = X_train
    fold_dict['X_val'] = X_val
    fold_dict['y_train_centered'] = y_train_centered

    # if we have snowfall as a predictor, first fit an OLS with only CLA as predictor
    # CLA is highly predictive of snowfall, in linear fashion..
    if fit_ols:
        linreg = LinearRegression(fit_intercept=False)
        x_cla_centered = X_train['cumulative_landfalling_area'] - X_train['cumulative_landfalling_area'].mean()
        fit = linreg.fit(X=pd.DataFrame(x_cla_centered), y=y_train_centered)
        resid = y_train_centered - fit.predict(pd.DataFrame(x_cla_centered))
        fold_dict['y_train_resid'] = resid # instead of y, use residuals for xgboost fitting

        # grab predictions of snowfall using lm on each validation set
        # should be fixed regardless of xgboost hyperparam, so best to do in advance
        val_CLA = pd.DataFrame(X_val['cumulative_landfalling_area'] - X_train['cumulative_landfalling_area'].mean())
        lm_val_preds = fit.predict(val_CLA)

        fold_dict['lm_val_preds'] = lm_val_preds

    data_CV_splits.append(fold_dict)

parallel_func = partial(process_hyperparam_chunk, 
                        etas=hyperparam_dict['etas'], 
                        booster=hyperparam_dict['booster'], 
                        tree_method=hyperparam_dict['tree_method'], 
                        nrounds=hyperparam_dict['nrounds'], 
                        early_stopping_rounds=hyperparam_dict['early_stopping_rounds'],
                        data_CV_splits=data_CV_splits,
                        shrink=args.shrink,
                        seed=XGB_SEED)

if __name__ == '__main__':

    with multiprocessing.Pool(processes=ncores) as pool:
        results_iterator = pool.imap_unordered(parallel_func, chunk_lst)
        print(f"Starting parallel processing of {len(chunk_lst)} chunks on {ncores} cores...")
        results = list(tqdm(results_iterator, total=len(chunk_lst)))
        print("Processing complete.")
    
    ols_avg_r2_shrunk = ols_pred(data_CV_splits, shrink=True)
    ols_avg_r2_noshrunk = ols_pred(data_CV_splits, shrink=False)
    
    full_df = pd.concat(results, ignore_index=True)
    full_df['test-r2-mean-ols-shrunk'] = ols_avg_r2_shrunk
    full_df['test-r2-mean-ols'] = ols_avg_r2_noshrunk
    full_df.to_csv(os.getcwd() + '/rounds/' + args.save_name)
