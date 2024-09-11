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

# load up the AR catalogs
curwd = os.getcwd()
catalog_paths = str(Path(curwd).parents[0]) + '/data/ar_catalogs/*.nc'
full_catalog = xr.open_mfdataset(catalog_paths)
# get rid of all non-antarctic points
catalog_subset = full_catalog.sel(lat=slice(-86, -39)).ar_binary_tag
# get rid of all time steps for which there is no AR present
is_ar_time = catalog_subset.any(dim = ['lat', 'lon'])
catalog_subset = catalog_subset.sel(time=is_ar_time)

# Load up the AIS mask
mask_path = str(Path(curwd).parents[0]) + '/data/antarctic_masks/AIS_Full_basins_Zwally_MERRA2grid_new.nc'
full_ais_mask = xr.open_dataset(mask_path).Zwallybasins > 0
# grab only points in the Southern Ocean area
ais_mask = full_ais_mask.sel(lat=slice(-86, -39))
# get ais points
ais_mask_lats = ais_mask.lat[np.where(ais_mask.to_numpy())[0]].to_numpy()
ais_mask_lons = ais_mask.lon[np.where(ais_mask.to_numpy())[1]].to_numpy()
ais_pts = set(zip(ais_mask_lats, ais_mask_lons))

########### CLUSTERING ###########

# hyperparameters
# arclength (km) on earth subtended by 1 radian  
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

    dataframe_year = obj_subset[obj_subset.time.dt.year == year]
    
    # construct dataarrays for each storm, organize into series
    dataarrays = dataframe_year.groupby('cluster').apply(construct_da, include_groups=False)
    dataarrays.name = 'data_array'

    # get series of whether each storm is landfalling or not
    dataframe_year['is_landfalling'] = dataframe_year.apply(lambda x: is_landfalling(x, ais_pts), axis=1)
    is_landfalling_cluster = dataframe_year.groupby('cluster')['is_landfalling'].any()

    # join into big ol dataframe and save
    dataframe = pd.concat([dataarrays, is_landfalling_cluster], axis=1)
    dataframe.to_pickle(f'/scratch/users/butlerj/extreme_antarctic_ars/dataframes/{year}_storm_df.pkl')

    # also convert to one-hot encoded raster format like Jonathan's catalogs
    year_times = catalog_subset.sel(time=(catalog_subset.time.dt.year == year)).time.to_numpy()
    da_lst = [None]*len(year_times)
    for i in range(len(year_times)):
        single_df = cluster_infos_df[cluster_infos_df.time == year_times[i]]
        n_storms = single_df.shape[0]
        storm_df = [None]*n_storms

        for j in range(single_df.shape[0]):
            lats = pd.Series(single_df[['cluster', 'lats', 'lons']].lats.iloc[j])
            lons = pd.Series(single_df[['cluster', 'lats', 'lons']].lons.iloc[j])

            points = pd.DataFrame({'lat':lats, 'lon':lons})
            points['cluster'] = single_df['cluster'].iloc[j]
            storm_df[j] = points
	    
        time_df = pd.concat(storm_df, axis=0)
        raster_day = time_df.set_index(['lat', 'lon']).to_xarray()
        da_lst[i] = raster_day

    year_da = xr.concat(da_lst, dim='time')
    year_da = year_da.assign_coords(time = year_times)
    augmented_times = np.array(pd.date_range(f'{year}-01-01T00:00:00.000000000', f'{year}-12-31T21:00:00.000000000', freq='3h'))
    year_da = year_da.reindex(lat=catalog_subset.lat, lon=catalog_subset.lon, time=augmented_times)

    year_da = year_da.fillna(0)
    year_da = year_da.cluster.astype(np.int16)
    year_da = year_da.chunk('auto')
    year_da.to_netcdf(f'/scratch/users/butlerj/extreme_antarctic_ars/datarrays/{year}_storm_da.nc')

    print(f'saved {year}')
