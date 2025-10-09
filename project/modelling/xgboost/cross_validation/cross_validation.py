import pandas as pd
import xarray as xr
import xgboost as xgb
import numpy as np
from itertools import product
from pathlib import Path
import os
import json
import argparse
import multiprocessing
from cv_utils import process_hyperparam_chunk

# parsing CV args to script
parser = argparse.ArgumentParser(description='Parser for CV arguments in hyperparameter search.')
parser.add_argument('--x_cols', nargs='+', required=True, help='A list of feature columns to use.')
parser.add_argument('--y_col', type=string, required=True, help='A string indicating target y variable.')
parser.add_argument('--hyperparam_json', type=string, required=True, help='File path to json with this round's CV hyperparameters.')
parser.add_argument('--chunk_size', type=int, default=100, help='How many hyperparam sets each worker will process in parallel.')
parser.add_argument('--save_name', type=string, required=True, help='Name of file with CV round results.')

load_path = Path(os.getcwd()).parents[0]/Path('dataset/datasets/model_ready/train.csv')

args = parser.parse_args()

with open('/rounds/' + args.hyperparam_json, 'r') as file:
    hyperparam_dict = json.load(file)
hyperparams_lst = [lst for key, lst in hyperparam_dict.items() if key in ['gammas', 'max_depth', 'lambdas', 'min_child_weights']]
hyperparams_lst = list(product(*hyperparams_lst))

# creating the chunks of hyperparameters
chunk_lst = []
chunk_size = args.chunk_size

for i in range(0, len(hyperparams_lst), chunk_size):
    chunk_lst.append(hyperparams_lst[i:i + chunk_size])

n_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

if __name__ == '__main__':
    with multiprocessing.Pool(processes=n_proc) as pool:
        results = pool.map(lambda x: process_hyperparam_chunk(x, 
                                                              hyperparam_dict['etas'], 
                                                              hyperparam_dict['booster'], 
                                                              hyperparam_dict['tree_method'], 
                                                              hyperparam_dict['n_rounds'], 
                                                              hyperparam_dict['early_stopping_rounds'], 
                                                              args.x_cols, 
                                                              args.y_col, 
                                                              load_path), chunk_lst)
    full_df = pd.concat(results, ignore_index=True)
    full_df.to_csv('/rounds/' + args.save_name)
