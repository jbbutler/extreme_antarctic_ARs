import os
from pathlib import Path
import numpy as np
import pandas as pd
import dask
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
import math
from tqdm import tqdm

import st_dbscan as st

from utils import arctan
from utils import average_angle
from utils import retrieve_neighbors
from utils import construct_da
from utils import is_landfalling
import utils

######## SETUP ########
scratch_dir = '/scratch/users/butlerj/extreme_antarctic_ars/'
catalog_subset = utils.load_catalogs()
ais_pts = utils.load_ais()

####### HYPERPARAM VARIATIONS #######
baseline_par_dict = {'seed':12345,
                     'eps_time':12,
                     'eps_space': 0.5,
                     'rep_pts': 10,
                     'min_pts': 5}

par_perturbations_dict = {'eps_time': [12, 14, 16, 18, 20, 22, 24],
                         'rep_pts': [15, 20, 50]}

# make the combos of parameters
combos = {}
combos['baseline'] = [baseline_par_dict]

for key in par_perturbations_dict.keys():
    
    n_combos = len(par_perturbations_dict[key])
    combos_lst = [None]*n_combos
    
    for i in range(n_combos):
        temp_dict = baseline_par_dict.copy()
        temp_dict[key] = par_perturbations_dict[key][i]
        combos_lst[i] = temp_dict
        
    combos[key] = combos_lst

synoptic_scale = 10**3
km_per_radian = 6.371*(10**3)

for key in combos.keys():
    
    key_dir = scratch_dir + 'hyperparam_analysis/' + key + '/'
    Path(key_dir).mkdir(parents=True, exist_ok=True)
    
    par_dicts = combos[key]
    
    for par_dict in par_dicts:
        
        if key == 'baseline':
            par_dir = key_dir
        else:
            par_dir = key_dir + key + f'_{par_dict[key]}/'
          
        Path(par_dir).mkdir(parents=True, exist_ok=True)
        
        # instantiating the clustering object
        cluster_obj = st.ST_DBSCAN(par_dict['eps_space']*synoptic_scale/km_per_radian, 
                                   par_dict['eps_space']*synoptic_scale/km_per_radian, 
                                   par_dict['eps_time']/24, 
                                   par_dict['min_pts'], 
                                   par_dict['min_pts'], 
                                   par_dict['rep_pts'])
        
        # doing the spatiotemporal clustering
        cluster_infos_df = cluster_obj.fit(catalog_subset)   
        # remove noise clusters
        obj_subset = cluster_infos_df[['cluster', 'lats', 'lons', 'time']]
        obj_subset = obj_subset[obj_subset['cluster'] != -1]
        
        dataframe = utils.construct_dataframe(obj_subset, ais_pts)
        dataframe.to_hdf(par_dir + '/storm_df.h5', key='df')
        da = utils.construct_dataarray(coord_dict=catalog_subset, big_df=obj_subset)
        da.to_netcdf(par_dir + '/storm_da.nc')
