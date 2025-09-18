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

import st_dbscan as st
import utils
from utils import arctan
from utils import average_angle
from utils import retrieve_neighbors
from utils import construct_da
from utils import is_landfalling

# parsing arguments submitted by user
parser = argparse.ArgumentParser(description="Parsing arguments to run the clustering algorithm.")
parser.add_argument("--synoptic_scale_frac", type=float, required=True, help="The fraction of a synoptic scale to use for the spatial epsilon.")
parser.add_argument("--eps_time_hours", type=float, required=True, help="The time epsilon to use in hours.")
parser.add_argument("--minpts", type=int, required=True, help="The min number of points to define a core point in DBSCAN.")
parser.add_argument("--n_rep_pts", type=int, required=True, help="The number of points to sample from each cluster in the spatial clustering step.")
parser.add_argument("--seed", type=int, default=12345, help="Random seed, for reproducibility's sake.")
parser.add_argument("--save_path", type=str, required=True, help="The path to the directory where you want to save the results.")

args = parser.parse_args()

####### SETUP #######
catalog_subset = utils.load_catalogs()
ais_pts = utils.load_ais()

########### CLUSTERING ###########

# hyperparameters
synoptic_scale = 10**3
km_per_radian = 6.371*(10**3) # arclength (km) on earth subtended by 1 radian

eps_space = args.synoptic_scale_frac*(synoptic_scale/km_per_radian) # converted to radians for Haversine metric
eps_space_1 = eps_space
eps_space_2 = eps_space
eps_time = args.eps_time_hours/24
minpts_1 = args.minpts
minpts_2 = args.minpts
n_rep_pts = args.n_rep_pts

# instantiating the clustering object
cluster_obj = st.ST_DBSCAN(eps_space_1, eps_space_2, eps_time, minpts_1, minpts_2, n_rep_pts)
# doing the spatiotemporal clustering
cluster_infos_df = cluster_obj.fit(catalog_subset)

########### POST-PROCESSING ###########
# remove noise clusters
obj_subset = cluster_infos_df[['cluster', 'lats', 'lons', 'time']]
obj_subset = obj_subset[obj_subset['cluster'] != -1]

dataframe = utils.construct_dataframe(obj_subset, ais_pts)
dataframe.to_hdf(args.save_path + f'/epsspace{args.synoptic_scale_frac}_epstime{args.eps_time_hours}_minpts{args.minpts}_nreppts{n_rep_pts}.h5', key='df')