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
import multiprocessing
from cv_utils import process_hyperparam_chunk
from functools import partial

# parsing CV args to script
parser = argparse.ArgumentParser(description='Parser for CV arguments in hyperparameter search.')
parser.add_argument('--x_cols', nargs='+', required=True, help='A list of feature columns to use.')
parser.add_argument('--y_col', type=str, required=True, help='A string indicating target y variable.')
parser.add_argument('--hyperparam_json', type=str, required=True, help='File path to json with this round\'s CV hyperparameters.')
parser.add_argument('--chunk_size', type=int, default=100, help='How many hyperparam sets each worker will process in parallel.')
parser.add_argument('--save_name', type=str, required=True, help='Name of file with CV round results.')
parser.add_argument('--ncores', type=int, help='How many cores to use in the parallelization.')

load_path = Path(os.getcwd()).parents[2]/Path('dataset/datasets/model_ready/train.csv')

args = parser.parse_args()

with open(os.getcwd() + '/rounds/' + args.hyperparam_json, 'r') as file:
    hyperparam_dict = json.load(file)
hyperparams_lst = [lst for key, lst in hyperparam_dict.items() if key in ['gammas', 'max_depth', 'lambdas', 'min_child_weights']]
hyperparams_lst = list(product(*hyperparams_lst))

# creating the chunks of hyperparameters
chunk_lst = []
chunk_size = args.chunk_size

for i in range(0, len(hyperparams_lst), chunk_size):
    chunk_lst.append(hyperparams_lst[i:i + chunk_size])
    
ncores = args.ncores
if ncores == None:
    ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

parallel_func = partial(process_hyperparam_chunk, 
                        etas=hyperparam_dict['etas'], 
                        booster=hyperparam_dict['booster'], 
                        tree_method=hyperparam_dict['tree_method'], 
                        nrounds=hyperparam_dict['nrounds'], 
                        early_stopping_rounds=hyperparam_dict['early_stopping_rounds'],
                        x_cols=args.x_cols, 
                        y_col=args.y_col, 
                        load_training_path=load_path)

if __name__ == '__main__':
    with multiprocessing.Pool(processes=ncores) as pool:
        results_iterator = pool.imap_unordered(parallel_func, chunk_lst)
        print(f"Starting parallel processing of {len(chunk_lst)} chunks on {ncores} cores...")
        results = list(tqdm(results_iterator, total=len(chunk_lst)))
        print("Processing complete.")
        
    full_df = pd.concat(results, ignore_index=True)
    full_df.to_csv(os.getcwd() + '/rounds/' + args.save_name)