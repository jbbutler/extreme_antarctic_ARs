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

from utils import arctan
from utils import average_angle
from utils import retrieve_neighbors

import pygmt

# load up the AR catalogs
curwd = os.getcwd()
catalog_paths = str(Path(curwd).parents[0]) + '/data/ar_catalogs/*.nc'
full_catalog = xr.open_mfdataset(catalog_paths)

# Load up the AIS mask
mask_path = str(Path(curwd).parents[0]) + '/data/antarctic_masks/AIS_Full_basins_Zwally_MERRA2grid_new.nc'
full_ais_mask = xr.open_dataset(mask_path).Zwallybasins > 0

# get rid of all non-antarctic points
catalog = full_catalog.sel(lat=slice(-86, -39))
ais_mask = full_ais_mask.sel(lat=slice(-86, -39))

catalog_years = np.unique(catalog.time.dt.year)

for year in catalog_years:

    catalog_subset = catalog.sel(time=(catalog.time.dt.year==year))
    # times to loop through to cluster
    times = catalog_subset.time.to_numpy()
    # instantiate empty list with each index corresponding to a time step
    # each entry will consist of a dataframe of clusters subdivided at that particular time
    cluster_infos = [None]*len(times)
    # from each cluster created, number of representative points to randomly sample
    n_rep_pts = 10

    ########### SPATIAL CLUSTERING ###########

    for i in range(len(times)):
    
        time_slice = catalog_subset.sel(time = times[i])
        # find lats/lons of AR points in this time step
        inds = np.argwhere(time_slice.to_numpy() == 1)
        storm_lats = time_slice.lat[inds[:,0]]
        storm_lons = time_slice.lon[inds[:,1]]

        # cluster spatially using DBSCAN, synoptic scale neighborhood size
        synoptic_scale = (10**3)/2
        km_per_radian = 6.371*(10**3) # arclength (km) on earth subtended by 1 radian  
        eps = synoptic_scale/km_per_radian # converted to radians for Haversine metric
        clustering = DBSCAN(eps=eps, min_samples=5, metric='haversine').fit_predict(X=np.column_stack((np.radians(storm_lats), np.radians(storm_lons))))
        fixed_time_df = pd.DataFrame({'lats': storm_lats, 'lons': storm_lons, 'cluster': clustering})

        # get the average lat-lon of each cluster, according to average_angle function above
        avg_positions = pd.DataFrame(fixed_time_df.groupby('cluster').apply(average_angle, include_groups=False).to_list(), 
                                columns=['cluster', 'mean_lat', 'mean_lon'])

        # randomly sample n_rep_pts-many points (without replacement) from each cluster and store
        rep_pts = fixed_time_df.groupby('cluster', as_index=False)[['lats', 'lons']].agg(lambda x: list(np.random.choice(x, min(n_rep_pts, len(x)), replace=False)))
        rep_pts.rename(columns={'lats':'rep_lats', 'lons':'rep_lons'}, inplace=True)

        rep_pts_df = pd.merge(rep_pts, avg_positions, on='cluster')

        # aggregate ALL lats and lons for each cluster into lists as well
        cluster_info = fixed_time_df.groupby('cluster', as_index=False)[['lats', 'lons']].agg(list)

        # combine all this info into single dataframe
        cluster_info = pd.merge(cluster_info, rep_pts_df, on='cluster')
        # add time into column
        cluster_info['time'] = times[i]
        # add this time-specific df into list of dfs
        cluster_infos[i] = cluster_info

    # stitch list of dataframes across time into big dataframe
    cluster_infos_df = pd.concat(cluster_infos, axis=0)
    cluster_infos_df.reset_index(drop=True, inplace=True)
    # preallocate empty column for cluster labels for each AR timestep
    ar_pt_df = cluster_infos_df[['mean_lat', 'mean_lon', 'rep_lats', 'rep_lons', 'time']]
    ar_pt_df['cluster'] = np.full(cluster_infos_df.shape[0], np.nan)

    ########### SPATIOTEMPORAL CLUSTERING ###########

    # loop through the dataframe made and unpack all of the representative points sampled for each cluster
    # resulting dataframe will have rows consisting of representative storm point
    # this is basically getting the data in the right format to do the ST-DBSCAN step
    unpacked_indices = []
    unpacked_lats = []
    unpacked_lons = []
    unpacked_times = []

    for index in list(ar_pt_df.index):
        num_pts = len(ar_pt_df.loc[index].rep_lats) + 1
        unpacked_indices = unpacked_indices + [index]*num_pts
        unpacked_times = unpacked_times + [ar_pt_df.loc[index].time]*num_pts
        unpacked_lats = unpacked_lats + list(np.radians(ar_pt_df.loc[index].rep_lats)) + [np.radians(ar_pt_df.loc[index].mean_lat)]
        unpacked_lons = unpacked_lons + list(np.radians(ar_pt_df.loc[index].rep_lons)) + [np.radians(ar_pt_df.loc[index].mean_lon)]

    unpacked_df = pd.DataFrame({'cluster':np.full(len(unpacked_indices), np.nan), 'space_cluster':unpacked_indices, 'time':unpacked_times, 'lat':unpacked_lats, 'lon':unpacked_lons})

    # hyperparams to ST-DBSCAN algorithm
    min_pts = 5
    noise_label = 7777777
    cluster_label = 0

    # same spatial scale as the spatial clustering
    synoptic_scale = (10**3)/2
    km_per_radian = 6.371*(10**3) # arclength (km) on earth subtended by 1 radian  
    eps_space = synoptic_scale/km_per_radian

    # neighboring points only considered within 18 hours of current point
    eps_time = 18/24

    ar_pt_df = unpacked_df

    # for each point
    for i in range(ar_pt_df.shape[0]):

        cur_obj = ar_pt_df.iloc[[i]]
        # if either unclustered or noise
        if math.isnan(ar_pt_df.loc[i, 'cluster']) or ar_pt_df.loc[i, 'cluster'] == noise_label:
        
            neighbors = retrieve_neighbors(cur_obj, ar_pt_df, eps_space, eps_time)
            # if less than min_pts neighbors, accounting for the point itself
            if neighbors.shape[0] < min_pts + 1:
                ar_pt_df.loc[i, 'cluster'] = noise_label
            # otherwise, start new cluster and label your neighbors accordingly
            else:
                cluster_label = cluster_label + 1
                ar_pt_df.loc[neighbors.index, 'cluster'] = cluster_label
                # indices to keep track of which points will be in cluster
                cluster_inds = list(neighbors.drop(i).index)

                # while we still have unprocessed cluster points
                while cluster_inds:
                    new_cur_obj = ar_pt_df.loc[[cluster_inds.pop()]]
                    new_neighbors = retrieve_neighbors(new_cur_obj, ar_pt_df, eps_space, eps_time)
                
                    if new_neighbors.shape[0] >= min_pts + 1:
                        # if neighboring point unlabelled, endow with cluster label
                        unlabelled = new_neighbors.loc[new_neighbors['cluster'].isnull()]
                        ar_pt_df.loc[unlabelled.index, 'cluster'] = cluster_label
                        # add these newly clustered points to list of unprocessed cluster points
                        cluster_inds = cluster_inds + list(unlabelled.index)

    cluster_assignments = ar_pt_df.groupby('space_cluster')['cluster'].apply(lambda series: series.value_counts().idxmax())
    # add cluster membership column back to original df
    cluster_infos_df['cluster'] = cluster_assignments

    # remove noise clusters
    # noise cluster meaning they are one-off ARs
    cluster_infos_df = cluster_infos_df[cluster_infos_df['cluster'] != noise_label]

    # get ais points
    ais_mask_lats = ais_mask.lat[np.where(ais_mask.to_numpy())[0]].to_numpy()
    ais_mask_lons = ais_mask.lon[np.where(ais_mask.to_numpy())[1]].to_numpy()
    ais_pts = set(zip(ais_mask_lats, ais_mask_lons))

    # determine which steps are landfalling and which are not, given a row of dataframe
    def is_landfalling(row):
        lats = np.array(np.degrees(row.lats))
        lons = np.array(np.degrees(row.lons))

        storm_pts = set(zip(lats, lons))

        return(bool(storm_pts & ais_pts))

    # add is_landfalling column to each AR at each time step
    cluster_infos_df['is_landfalling'] = cluster_infos_df.apply(is_landfalling, axis=1)

    # save the dataframe format
    cluster_infos_df.to_pickle(str(Path(curwd).parents[0]) + f'/output/dataframes/cluster_infos_{year}.pkl')

    # also convert to one-hot encoded raster format like Jonathan's catalogs
    da_lst = [None]*times.shape[0]
    for i in range(times.shape[0]):
        single_df = cluster_infos_df[cluster_infos_df.time == times[i]]
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
    year_da = year_da.assign_coords(time = times)
    augmented_times = np.array(pd.date_range(f'{year}-01-01T00:00:00.000000000', f'{year}-12-31T21:00:00.000000000', freq='3h'))
    year_da = year_da.reindex(lat=catalog_subset.lat, lon=catalog_subset.lon, time=augmented_times)

    year_da = year_da.fillna(0)
    year_da = year_da.chunk('auto')
    year_da.to_netcdf(path=str(Path(curwd).parents[0]) + f'/output/datarrays/{year}_clusters.nc')