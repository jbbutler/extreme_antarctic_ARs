# Script to run the clustering algorithm for the year 2022, to be used with
# the accompanying Jupyter Book for my AGU24 presentation

import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
import math
from tqdm import tqdm

import pdb

import st_dbscan_tutorial as st

import utils_tutorial as utils
from utils_tutorial import arctan
from utils_tutorial import average_angle
from utils_tutorial import retrieve_neighbors
from utils_tutorial import construct_da
from utils_tutorial import is_landfalling

####### SETUP #######
catalog_subset = utils.load_catalogs()
catalog_subset = catalog_subset.sel(time = catalog_subset.time.dt.year == 2022)
ais_pts = utils.load_ais()

########### CLUSTERING ###########

# hyperparameters  
synoptic_scale = 10**3
km_per_radian = 6.371*(10**3) # arclength (km) on earth subtended by 1 radian

eps_space = synoptic_scale/(2*km_per_radian) # converted to radians for Haversine metric
eps_space_1 = eps_space
eps_space_2 = eps_space
eps_time = 18/24
minpts_1 = 5
minpts_2 = 5
n_rep_pts = 10

# instantiating the clustering object
cluster_obj = st.ST_DBSCAN(eps_space_1, eps_space_2, eps_time, minpts_1, minpts_2, n_rep_pts)
# doing the spatiotemporal clustering
cluster_infos_df = cluster_obj.fit(catalog_subset)

########### POST-PROCESSING ###########
# remove noise clusters
obj_subset = cluster_infos_df[['cluster', 'lats', 'lons', 'time']]
obj_subset = obj_subset[obj_subset['cluster'] != -1]

# save the dataframes and the xarray objects by year
catalog_years = np.unique(catalog_subset.time.dt.year)

for year in catalog_years:

    # grab the dataframe of cluster results for each time step, but only for one year
    dataframe_year = obj_subset[obj_subset.time.dt.year == year]
    # construct and save the dataframe for that year
    dataframe = utils.construct_dataframe(dataframe_year, ais_pts)
    dataframe.to_hdf(f'/global/u1/j/jbbutler/extreme_antarctic_ARs/data/ar_database/tutorial_df/{year}_storm_df.h5', key='df')

    # construct and save the one-hot-encoded data array format for that year
    #coord_dict = {'lats': catalog_subset.lat, 'lons': catalog_subset.lon}
    #year_da = utils.construct_dataarray(dataframe_year, coord_dict)
    #year_da.to_netcdf(f'/scratch/users/butlerj/extreme_antarctic_ars/datarrays/{year}_storm_da.nc')

    print(f'saved {year}')
