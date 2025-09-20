# File to run the clustering algorithm on the Wille 2021 catalog, with a particular set of hyperparameters
#
# Jimmy Butler
# September 2025

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import dask
import xarray as xr
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
import math
from tqdm import tqdm
import random

import dask.multiprocessing
dask.config.set(scheduler='processes', num_workers = 10)

import st_dbscan as st
import utils
from utils import arctan
from utils import average_angle
from utils import retrieve_neighbors
from utils import construct_da
from utils import is_landfalling

# parsing arguments submitted by user, 
parser = argparse.ArgumentParser(description="Parsing arguments to run the clustering algorithm.")
parser.add_argument("--synoptic_scale_frac", type=float, default=0.5, help="The fraction of a synoptic scale to use for the spatial epsilon.")
parser.add_argument("--eps_time_hours", type=float, default=12, help="The time epsilon to use in hours.")
parser.add_argument("--minpts", type=int, default=5, help="The min number of points to define a core point in DBSCAN.")
parser.add_argument("--n_rep_pts", type=int, default=10, help="The number of points to sample from each cluster in the spatial clustering step.")
parser.add_argument("--seed", type=int, default=12345, help="Random seed, for reproducibility's sake.")
parser.add_argument("--save_path", type=str, required=True, help="The path to the directory where you want to save the results.")

# optional arguments. if the user supplies perturbations, then the clustering algorithm will be run on these individual perturbations.
parser.add_argument("--frac_perturbs", type=float, nargs='*', default=[], help="A list of perturbations of the synoptic scale fraction.")
parser.add_argument("--time_perturbs", type = float, nargs='*', default=[], help="A list of perturbations of the epsilon time value.")
parser.add_argument("--minpts_perturbs", type=int, nargs='*', default=[], help="A list of perturbations of the minpts value.")
parser.add_argument("--reppts_perturbs", type=int, nargs='*', default=[], help="A list of perturbations of reppts.")
parser.add_argument("--seed_perturbs", type=int, nargs='*', default=[], help="A list of perturbatinos of the random seed.")

args = parser.parse_args()

# combining into a dictionary of parameters to then loop through
baseline_par_dict = {'seed':args.seed,
                     'eps_time':args.eps_time_hours,
                     'eps_space': args.synoptic_scale_frac,
                     'rep_pts': args.n_rep_pts,
                     'min_pts': args.minpts}

par_perturbations_dict = {'eps_time': args.time_perturbs,
                         'rep_pts': args.reppts_perturbs,
                         'eps_space': args.frac_perturbs,
                         'min_pts': args.minpts_perturbs,
                         'seed': args.seed_perturbs}

# make the combos of parameters
combos = []
combos.append(baseline_par_dict)

for key in par_perturbations_dict.keys():
    n_combos = len(par_perturbations_dict[key])
    for i in range(n_combos):
        temp_dict = baseline_par_dict.copy()
        temp_dict[key] = par_perturbations_dict[key][i]
        combos.append(temp_dict)


########### CLUSTERING ###########

def code(i):

    catalog_subset = utils.load_catalogs()
    ais_pts = utils.load_ais()
    
    par_dict = combos[i]
    random.seed(par_dict['seed'])
    # hyperparameters
    synoptic_scale = 10**3
    km_per_radian = 6.371*(10**3) # arclength (km) on earth subtended by 1 radian

    eps_space = par_dict['eps_space']*(synoptic_scale/km_per_radian) # converted to radians for Haversine metric
    eps_space_1 = eps_space
    eps_space_2 = eps_space
    eps_time = par_dict['eps_time']/24
    minpts_1 = par_dict['min_pts']
    minpts_2 = par_dict['min_pts']
    n_rep_pts = par_dict['rep_pts']

    # instantiating the clustering object
    cluster_obj = st.ST_DBSCAN(eps_space_1, eps_space_2, eps_time, minpts_1, minpts_2, n_rep_pts)
    # doing the spatiotemporal clustering
    cluster_infos_df = cluster_obj.fit(catalog_subset)

    ########### POST-PROCESSING ###########
    # remove noise clusters
    obj_subset = cluster_infos_df[['cluster', 'lats', 'lons', 'time']]
    obj_subset = obj_subset[obj_subset['cluster'] != -1]

    dataframe = utils.construct_dataframe(obj_subset, ais_pts)
    dataframe.to_hdf(args.save_path + f'/epsspace{par_dict['eps_space']}_epstime{par_dict['eps_time']}_minpts{par_dict['min_pts']}_nreppts{par_dict['rep_pts']}_seed{par_dict['seed']}.h5', key='df')

futures = [dask.delayed(code)(i) for i in range(len(combos))]
results = dask.compute(futures)
