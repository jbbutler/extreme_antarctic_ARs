from sklearn.metrics.pairwise import haversine_distances
import numpy as np
import pandas as pd
import xarray as xr
import os
from pathlib import Path

# homegrown arctan function to make sure that, for a given x and y, the
# the angle corresponds to the correct half of the unit circle
def arctan(x, y):
    if y/x > 0:
        if x > 0:
            return(np.arctan(y/x))
        else:
            return(np.arctan(y/x)-np.pi)
    else:
        if x > 0:
            return(np.arctan(y/x))
        else:
            return(np.pi+np.arctan(y/x))
    

# following wikipedia article on circular mean
# standard arithmetic means of non-euclidean (i.e. cyclic) spaces can behave badly
# instead, convert angles to unit vectors in R3, average components, find angles of avg. vector
def average_angle(subdf):
    lats = np.radians(subdf.lats)
    lons = np.radians(subdf.lons)

    x = np.cos(lats)*np.cos(lons)
    y = np.cos(lats)*np.sin(lons)
    z = np.sin(lats)
    
    avg_x = np.mean(x)
    avg_y = np.mean(y)
    avg_z = np.mean(z)

    avg_lat = np.arcsin(avg_z)
    avg_lon = arctan(avg_x, avg_y)

    return (subdf.name, np.degrees(avg_lat), np.degrees(avg_lon))


def retrieve_neighbors(object, data, eps_space, eps_time):
    '''
    object: a single row of dataframe with time, mean_lat, mean_lon columns;
        represents one of possibly many ARs at a single time step
    data: the rest of the dataset to cluster
    eps_space: neighborhood size in space (angular size)
    eps_time: neighborohod size in time (in days)
    '''
    
    obj_time = object['time'].iloc[0]
    obj_loc = object[['lat', 'lon']]

    # find neighbors in time
    time_neighbors = data.loc[np.abs((data['time'] - obj_time).dt.total_seconds()/86400) <= eps_time]
    # among time neighbors, find space neighbors
    st_neighbors = time_neighbors.loc[haversine_distances(time_neighbors[['lat', 'lon']], obj_loc) <= eps_space]

    return(st_neighbors)

# determine which steps are landfalling and which are not, given a row of dataframe
def is_landfalling(row, ais_pts):
    lats = np.array(row.lats)
    lons = np.array(row.lons)

    storm_pts = set(zip(lats, lons))

    return(bool(storm_pts & ais_pts))

# construct the cluster-specific dataarrays from the cluster dataframes
def construct_da(cluster_df):

    n_time_pts = cluster_df.shape[0]
    dfs = [None]*n_time_pts

    for i in range(n_time_pts):
    
        lats = cluster_df.iloc[i].lats
        lons = cluster_df.iloc[i].lons
        storm_df = pd.DataFrame({'lats': lats, 'lons': lons})
        storm_df['time'] = cluster_df.iloc[i].time
    
        dfs[i] = storm_df
    
    storm_df = pd.concat(dfs, axis=0)
    storm_df['clust'] = 1

    storm_df = storm_df.set_index(['time', 'lats', 'lons'])
    storm_da = storm_df.to_xarray()
    storm_da = storm_da.fillna(0).astype(np.int8).clust
    
    return storm_da

def load_catalogs(years=None):

    curwd = os.getcwd()
    catalog_paths = str(Path(curwd).parents[0]) + '/data/ar_catalogs/*.nc'
    full_catalog = xr.open_mfdataset(catalog_paths)

    if years is not None:
        full_catalog = full_catalog.sel(time=full_catalog.time.dt.year.isin(years))

    # get rid of all non-antarctic points
    catalog_subset = full_catalog.sel(lat=slice(-86, -39)).ar_binary_tag
    # get rid of all time steps for which there is no AR present
    is_ar_time = catalog_subset.any(dim = ['lat', 'lon'])
    catalog_subset = catalog_subset.sel(time=is_ar_time)

    return catalog_subset

def load_ais():

    curwd = os.getcwd()
    # Load up the AIS mask
    mask_path = str(Path(curwd).parents[0]) + '/data/antarctic_masks/AIS_Full_basins_Zwally_MERRA2grid_new.nc'
    full_ais_mask = xr.open_dataset(mask_path).Zwallybasins > 0
    # grab only points in the Southern Ocean area
    ais_mask = full_ais_mask.sel(lat=slice(-86, -39))
    # get ais points
    ais_mask_lats = ais_mask.lat[np.where(ais_mask.to_numpy())[0]].to_numpy()
    ais_mask_lons = ais_mask.lon[np.where(ais_mask.to_numpy())[1]].to_numpy()
    ais_pts = set(zip(ais_mask_lats, ais_mask_lons))

    return ais_pts

def construct_dataframe(big_df, ais_pts):

    # construct dataarrays for each storm, organize into series
    dataarrays = big_df.groupby('cluster').apply(construct_da, include_groups=False)
    dataarrays.name = 'data_array'

    # get series of whether each storm is landfalling or not
    big_df['is_landfalling'] = big_df.apply(lambda x: is_landfalling(x, ais_pts), axis=1)
    is_landfalling_cluster = big_df.groupby('cluster')['is_landfalling'].any()
    df = pd.concat([dataarrays, is_landfalling_cluster], axis=1)

    return df

def construct_dataarray(big_df, coord_dict):
    # coord_dict is dictionary of lats, lons: the coordinates you would like to set for the resulting data array
    # big_df is the dataframe output of the clustering (each row is AR at particular point in time)

    # also convert to one-hot encoded raster format like Jonathan's catalogs
    times = np.sort(big_df.time.unique())
    da_lst = [None]*len(times)
    for i in range(len(times)):
        single_df = big_df[big_df.time == times[i]]
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

    da = xr.concat(da_lst, dim='time')
    da = da.assign_coords(time = times)

    # fill in times for which there was no AR originally
    years = big_df.time.dt.year.unique()
    augmented_times_tot = [None]*len(years)
    for i, year in enumerate(years):
        augmented_times_tot[i] = np.array(pd.date_range(f'{year}-01-01T00:00:00.000000000', f'{year}-12-31T21:00:00.000000000', freq='3h'))
    
    augmented_times = np.concatenate(augmented_times_tot)

    da = da.reindex(lat=coord_dict['lats'], lon=coord_dict['lons'], time=augmented_times)

    da = da.fillna(0)
    da = da.cluster.astype(np.int16)
    da = da.chunk('auto')

    return da