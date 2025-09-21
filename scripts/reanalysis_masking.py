# Script that will fill in the dataframe of AR events with relevant information about each storm, like durations, landfalling regions
# as well as masking re-analysis data to obtain quantities like snowfall, moisture content, energy, etc.
#
# Jimmy Butler
# June 2025
import pandas as pd
import xarray as xr
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import xarray as xr
from tqdm import tqdm
import dask
import utils

# configure paths to where each MERRA-2 dataset data is stored
# ideally in the future, this could just be replaced with masking data streamed directly
home_dir = str(Path(os.getcwd()).parents[1])
scratch_path = '/pscratch/sd/j/jbbutler/'
inst1_data_path = '/pscratch/sd/j/jbbutler/merra2_data_T2m_V10m_SLP_IWV/'
tavg1_precip_data_path = '/pscratch/sd/j/jbbutler/merra2_data_precip_ivt/'
tavg1_850hPa_wind_data_path = '/pscratch/sd/j/jbbutler/merra2_data_850hPa_wind/'
tavg1_omega500_data_path = '/pscratch/sd/j/jbbutler/merra2_data_omega500/'


# load up all of the dataframes
df_path = home_dir + '/data/catalog/...'
dataframe = pd.read_hdf(df_path)

# only take those that are landfalling
landfalling_storms = dataframe[dataframe.is_landfalling]

# load up the cell areas dataarray
cell_areas = xr.open_dataset('~/extreme_antarctic_ARs/data/area/MERRA2_gridarea.nc')
cell_areas = cell_areas.cell_area
cell_areas = cell_areas.assign_coords(lat=cell_areas.lat.round(5), lon=cell_areas.lon.round(5))
# load up the antarctic ice sheet mask
ais_mask = xr.open_dataset('~/extreme_antarctic_ARs/data/antarctic_masks/AIS_Full_basins_Zwally_MERRA2grid_new.nc')
ais_mask = ais_mask > 0
ais_mask = ais_mask.assign_coords(lat=ais_mask.lat.round(5), lon=ais_mask.lon.round(5)) # this is to avoid issues with -0 not matching 0
# compute the climatologies (so far, only for SLP, 2m-temperature, and IWV/TQV)
monthly_averages = xr.open_mfdataset('/pscratch/sd/j/jbbutler/merra2_monthly_data/*.nc4')
climatology_ds = monthly_averages.groupby(monthly_averages.time.dt.month).mean().compute()

################################### Functions to compute areal/durational quantities #################################
def compute_max_area(ar_da, ais_da=None):
    '''
    A function that, given a binary mask DataArray for a storm, computes the max area occupied over lifetime

    Inputs:
        ar_da (xarray.DataArray): the binary valued DataArray for that storm
        ais_da (xarray.DataArray): if provided, find max area occupied over AIS (default: over AIS + ocean)
    Outputs:
        max_area (float): the area in km^2
    '''

    # just to be safe, round coordinates so that -0 matches with 0 degrees, in case that appears
    ar_da_rounded = ar_da.assign_coords(lat=ar_da.lat.round(5), lon=ar_da.lon.round(5))
    
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=ar_da_rounded.lat, lon=ar_da_rounded.lon).Zwallybasins
        storm_da_subset = ar_da_rounded.where(storm_ais_mask, 0)
    else:
        storm_da_subset = ar_da_rounded.copy()
    
    grid_area_storm = cell_areas.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)
    max_area = float(storm_da_subset.dot(grid_area_storm).max().values/(1000**2))
    
    return max_area

def compute_max_southward_extent(ar_da):
    '''
    A function that, given a binary mask DataArray for a storm, computes the lowest latitude it occupied

    Inputs:
        ar_da (xarray.DataArray): the binary valued DataArray for that storm
    Outputs:
        (float): the lowest latitude in degrees
    '''
    ar_da_rounded = ar_da.assign_coords(lat=ar_da.lat.round(5), lon=ar_da.lon.round(5))
    return np.min(ar_da.lat.values)

def compute_mean_area(ar_da, ais_da=None):
    '''
    A function that, given a binary mask DataArray for a storm, computes the mean area occupied over lifetime

    Inputs:
        ar_da (xarray.DataArray): the binary valued DataArray for that storm
        ais_da (xarray.DataArray): if provided, find mean area occupied over AIS (default: over AIS + ocean)
    Outputs:
        mean_area (float): the area in km^2
    '''
    
    ar_da_rounded = ar_da.assign_coords(lat=ar_da.lat.round(5), lon=ar_da.lon.round(5))
    
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=ar_da_rounded.lat, lon=ar_da_rounded.lon).Zwallybasins
        storm_da_subset = ar_da_rounded.where(storm_ais_mask, 0)
    else:
        storm_da_subset = ar_da_rounded.copy()
    
    grid_area_storm = cell_areas.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)
    mean_area = float(storm_da_subset.dot(grid_area_storm).mean().values/(1000**2))
    return mean_area

def compute_cumulative_spacetime(ar_da, ais_da=None):
    '''
    A function that, given a binary mask DataArray for a storm, computes the cumulative amount
        of space and time the storm spent over the AIS (measured in km^2 x days)

    Inputs:
        ar_da (xarray.DataArray): the binary valued DataArray for that storm
        ais_da (xarray.DataArray): if provided, find quantity occupied over AIS (default: over AIS + ocean)
    Outputs:
        cumulative_area (float): the cumulative area in km^2 x days
    '''
    
    ar_da_rounded = ar_da.assign_coords(lat=ar_da.lat.round(5), lon=ar_da.lon.round(5))
    
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=ar_da_rounded.lat, lon=ar_da_rounded.lon).Zwallybasins
        storm_da_subset = ar_da_rounded.where(storm_ais_mask, 0)
    else:
        storm_da_subset = ar_da_rounded.copy()
    
    grid_area_storm = cell_areas.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)
    cumulative_area = float((3*storm_da_subset.dot(grid_area_storm)).sum().values/((1000**2)*24)) #3 comes from 3 hourly blocks of Wille catalog
    return cumulative_area

def compute_duration(ar_da):
    '''
    Returns the duration of a storm. Note if the storm only occupies one 3 hourly time step, that storm's duration is 3 hours.

    Inputs:
        ar_da (xarray.DataArray): the binary mask for the storm
    Outputs:
        days (np.timedelta64): number of hours
    '''
    days = (ar_da.time.max() - ar_da.time.min()).values.astype('timedelta64[h]').astype(int) + np.timedelta64(3, 'h')
    return days

def add_start_date(ar_da):
    '''
    Returns the start date of a storm.

    Inputs:
        ar_da (xarray.DataArray): the binary mask for the storm
    Outputs:
        start (np.datetime64): starting time (3 hourly level)
    '''
    start = ar_da.time.min().values
    return start

def add_end_date(ar_da):
    '''
    Returns the end date of a storm.

    Inputs:
        ar_da (xarray.DataArray): the binary mask for the storm
    Outputs:
        end (np.datetime64): ending time (3 hourly level)
    '''
    end = ar_da.time.max().values
    return end

def find_landfalling_region(ar_da, region_masks):
    '''
    Finding the region in which the storm makes landfall. Regions provided by David Mikolajczyk
    at UW Madison, with additional regions for the RIS, FRIS, and completing the connection between
    QMC and END. Landfalling region determined as which one it has the most CLA over.

    Inputs:
        ar_da (xarray.DataArray): the binary mask DataArray
        region_masks (dictionary): keys are strings indicating region names, values are xarray.DataArray
            masks indicating pixels for those regions
    Outputs:
        winning_region (string): the string name of the region it spends most area x time over
    '''

    region_CLA = {}
    for label, mask in region_masks.items():
        region_CLA[label] = compute_cumulative_spacetime(ar_da, ais_da=mask)

    region_CLA = pd.Series(region_CLA)
    winning_region = region_CLA.idxmax()

    return winning_region

def find_region_masks(region_defs, ais_da):
    '''
    Helper function for the above find_landfalling_region function. Given longitude bounds for each region,
    find binary masks for each of these from the AIS mask
    
    Inputs:
        region_defs (dictionary): dictionary whose keys are strings indicating names of regions and
            values are lists with lower and upper longitude bounds for the regions
        ais_da (xarray.DataArray): the binary mask for the AIS
    Output:
        region_masks (dictionary): keys are names of regions (strings), and values are DataArray masks for
            just that region
    '''

    region_masks = {}

    for label, bound in region_defs.items():
        region_masks[label] = ais_da.where((ais_da.lon > bound[0]) & (ais_da.lon < bound[1]), False)

    return region_masks

def extract_trajectory(ar_da):
    '''
    Given an AR's binary valued DataArray, return a curve representing the path of the AR through space and time

    Inputs:
        ar_da (xarray.DataArray): the binary valued DataArray
    Output:
        trajectory_df (pd.DataFrame): a DataFrame where each row is the average lat/lon of the storm at a particular time.
    '''
    times = ar_da.time.values

    avg_lons = []
    avg_lats = []

    for time in times:
        
        time_slice = ar_da.sel(time=time)
        inds = np.argwhere(time_slice.values == 1)
        storm_lats = time_slice.lat[inds[:,0]]
        storm_lons = time_slice.lon[inds[:,1]]

        time_slice_coords = pd.DataFrame({'lats':storm_lats, 'lons':storm_lons})
        time_slice_coords.name = '1'
        
        avg_angle = utils.average_angle(time_slice_coords)

        avg_lons.append(avg_angle[2])
        avg_lats.append(avg_angle[1])

    trajectory_df = pd.DataFrame({'time': times, 'avg_lon': avg_lons, 'avg_lat': avg_lats})

    return trajectory_df

########################### Functions to compute aggregates of MERRA-2 data #############################
def compute_cumulative(storm_da, var_da, area_da, ais_da=None):
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
        storm_da_subset = storm_da.where(storm_ais_mask, 0)
    else:
        storm_da_subset = storm_da.copy()

    var_da_subset = var_da.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)

    storm_cell_areas = area_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    amt_per_3hr = storm_cell_areas.dot((storm_da_subset*var_da_subset))
    cumulative_storm_val = float((amt_per_3hr).sum())

    return cumulative_storm_val

def compute_max_intensity(storm_da, var_da, area_da, ais_da=None):
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
        storm_da_subset = storm_da.where(storm_ais_mask, 0)
    else:
        storm_da_subset = storm_da.copy()
        
    var_da_subset = var_da.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)
    max_intensity_val = float((storm_da_subset*var_da_subset).max())

    return max_intensity_val

def compute_min_SLP(storm_da, var_da, area_da, ais_da):
    
    storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
    storm_ocean_mask = np.logical_not(storm_ais_mask)
    storm_da_ais = storm_da.where(storm_ais_mask, 0)
    storm_da_ocean = storm_da.where(storm_ocean_mask, 0)
    
    var_da_subset = var_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    first_landfall = np.min(storm_da.time[storm_da_ais.any(dim=['lat', 'lon'])].values)
      
    first_day = (storm_da_ocean*var_da_subset).sel(time=first_landfall).values
    min_slp = np.min(first_day[first_day > 0], initial=99999999)

    return min_slp

def compute_max_SLPgrad(storm_da, var_da, area_da, ais_da):

    storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
    storm_ocean_mask = np.logical_not(storm_ais_mask)
    storm_da_ais = storm_da.where(storm_ais_mask, 0)
    storm_da_ocean = storm_da.where(storm_ocean_mask, 0)

    var_da_subset = var_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    first_landfall = np.min(storm_da.time[storm_da_ais.any(dim=['lat', 'lon'])].values)

    var_da_subset_landfall = var_da_subset.sel(time=first_landfall)
    storm_da_ocean_landfall = storm_da_ocean.sel(time=first_landfall)

    # return -1 if no points over the ocean at first landfalling time
    if (storm_da_ocean_landfall == 0).all().values:
        return -1
    ## compute pressure gradient
    # convert to radians
    rads = var_da_subset_landfall.assign_coords(lon=np.radians(var_da_subset_landfall.lon), lat=np.radians(var_da_subset_landfall.lat))
    # partials in the latitude direction (spherical coordinates)
    r = 6378 # radius of Earth in km
    lat_partials = rads.differentiate('lat')/r
    # partials in the longitudinal direction (spherical coordinates)
    lon_partials = rads.differentiate('lon')/(np.sin(rads.lat)*r)
    
    magnitude = np.sqrt(lon_partials**2 + lat_partials**2)
    max_grad = np.max(magnitude.values*storm_da_ocean_landfall.values)

    return max_grad

def compute_max_elevation_grad(storm_da, var_da, area_da, ais_da):

    storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
    storm_ocean_mask = np.logical_not(storm_ais_mask)
    storm_da_ais = storm_da.where(storm_ais_mask, 0)
    storm_da_ocean = storm_da.where(storm_ocean_mask, 0)

    var_da_subset = var_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    first_landfall = np.min(storm_da.time[storm_da_ais.any(dim=['lat', 'lon'])].values)

    var_da_subset_landfall = var_da_subset.sel(time=first_landfall)
    storm_da_ocean_landfall = storm_da_ocean.sel(time=first_landfall)

    # return -1 if no points over the ocean at first landfalling time
    if (storm_da_ocean_landfall == 0).all().values:
        return -1
    ## compute pressure gradient
    # convert to radians
    rads = var_da_subset_landfall.assign_coords(lon=np.radians(var_da_subset_landfall.lon), lat=np.radians(var_da_subset_landfall.lat))
    # partials in the latitude direction (spherical coordinates)
    r = 6378 # radius of Earth in km
    lat_partials = rads.differentiate('lat')/r
    # partials in the longitudinal direction (spherical coordinates)
    lon_partials = rads.differentiate('lon')/(np.sin(rads.lat)*r)
    
    magnitude = np.sqrt(lon_partials**2 + lat_partials**2)
    max_grad = np.max(magnitude.values*storm_da_ocean_landfall.values)

    return max_grad

def compute_max_landfalling_wind(storm_da, var_da, area_da, ais_da):

    storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
    storm_ocean_mask = np.logical_not(storm_ais_mask)
    storm_da_ais = storm_da.where(storm_ais_mask, 0)
    storm_da_ocean = storm_da.where(storm_ocean_mask, 0)
    first_landfall = np.min(storm_da.time[storm_da_ais.any(dim=['lat', 'lon'])].values)

    var_da_subset = var_da.sel(lat=storm_da.lat, lon=storm_da.lon)

    storm_da_ocean_landfall = storm_da_ocean.sel(time=first_landfall)
    if (storm_da_ocean_landfall == 0).all().values:
        return -1
    
    first_day = (storm_da_ocean*var_da_subset).sel(time=first_landfall).values
    # for some reason, 850hPa wind is null over the AIS. So, we take the max over all ocean
    # points, ignoring an NaNs
    max_wind = np.nanmax(first_day, initial=-999999)

    return max_wind

def compute_avg_landfalling_wind(storm_da, var_da, area_da, ais_da):

    storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
    storm_ocean_mask = np.logical_not(storm_ais_mask)
    storm_da_ais = storm_da.where(storm_ais_mask, 0)
    storm_da_ocean = storm_da.where(storm_ocean_mask, 0)
    first_landfall = np.min(storm_da.time[storm_da_ais.any(dim=['lat', 'lon'])].values)

    storm_da_ocean_landfall = storm_da_ocean.sel(time=first_landfall)
    if (storm_da_ocean_landfall == 0).all().values:
        return -1

    var_da_subset = var_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    # for some reason, 850hPa wind is null for values in and around the AIS
    # so, we compute over the ocean, where the 850hPa wind is also not null
    notnull = var_da_subset.notnull()
    # fill null locations with 0; will be excluded from computation anyway
    var_da_subset = var_da_subset.fillna(0)
    storm_da_ocean_notnull = storm_da_ocean*notnull
    storm_cell_areas = area_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    tot_area = storm_da_ocean_notnull.dot(storm_cell_areas)
    
    avg_wind = (storm_cell_areas.dot(storm_da_ocean_notnull*var_da_subset)/tot_area).sel(time=first_landfall).values

    return avg_wind

def compute_average(storm_da, var_da, area_da, ais_da=None):
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
        storm_da_subset = storm_da.where(storm_ais_mask, 0)
    else:
        storm_da_subset = storm_da.copy()

    var_da_subset = var_da.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)

    storm_cell_areas = area_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    tot_area = storm_da_subset.dot(storm_cell_areas)
    avg_storm_val = float((storm_cell_areas.dot(storm_da_subset*var_da_subset)/tot_area).mean())

    return avg_storm_val

def grab_MERRA2_files(storm_da, ticker):
    
    dates = np.unique(storm_da.time.dt.date.values)

    fnames = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        fname = ticker + '.' + date_str + '.nc4.nc4'
        fnames.append(fname)

    return fnames

def compute_raw_summaries(storm_da, func_vars_dict, cell_areas, ticker, data_path, ivt=False):
    
    storm_da = storm_da.assign_coords(lat=storm_da.lat.round(5), lon=storm_da.lon.round(5))
    fnames = grab_MERRA2_files(storm_da, ticker)
    
    var_lst = np.unique(np.array(list(func_vars_dict.keys()))[:,1])

    ds_lst = []
    for fname in fnames:
        ds = xr.open_dataset(data_path + fname)
        ds_lst.append(ds[var_lst].sel(time = ds.time.dt.hour % 3 == 0))

    obs_ds = xr.concat(ds_lst, dim='time')
    if ivt:
        obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5), time=obs_ds.time - np.timedelta64(30, 'm'))
    else:
        obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5))

    summaries = []
    for key, func in func_vars_dict.items():
        single_var_da = obs_ds[key[1]]
        summaries.append(func(storm_da, single_var_da, cell_areas))

    return summaries

def compute_anomaly_summaries(storm_da, func_vars_dict, climatology_dict, cell_areas, ticker, data_path):
    
    storm_da = storm_da.assign_coords(lat=storm_da.lat.round(5), lon=storm_da.lon.round(5))
    fnames = grab_MERRA2_files(storm_da, ticker)
    var_lst = np.unique(np.array(list(func_vars_dict.keys()))[:,1])
    
    ds_lst = []
    for fname in fnames:
        ds = xr.open_dataset(data_path + fname)
        ds_lst.append(ds[var_lst].sel(time = ds.time.dt.hour % 3 == 0))
        
    obs_ds = xr.concat(ds_lst, dim='time')
    obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5))
    
    summaries = []
    for key, func in func_vars_dict.items():
        actual_da = obs_ds[key[1]]
        climatology = climatology_dict[key[1]]
        climatology = climatology.assign_coords(lat=climatology.lat.round(5), lon=climatology.lon.round(5))
        single_var_da = xr.apply_ufunc(lambda da, clim: da-clim, actual_da.groupby('time.month'), climatology).drop_vars('month')
        summaries.append(func(storm_da, single_var_da, cell_areas))
        
    return summaries

# function to expand dataarray mask to include points which were within 24 hours of an AR point
# used to get a better assessment of precip due to an AR
def augment_storm_da(storm_da):
    
    start = storm_da.time.values[0]
    end = storm_da.time.values[-1] + np.timedelta64(1, 'D')
    full_dates=pd.date_range(start, end, freq='3h')
    
    unincluded_times = set(np.array(full_dates)) - set(storm_da.time.values)
    
    unincluded_array = np.zeros((len(unincluded_times), storm_da.shape[1], storm_da.shape[2]))
    unincluded_coords = {'time' : np.array(list(unincluded_times)), 'lat': storm_da.lat.values, 'lon': storm_da.lon.values}
    unincluded_da = xr.DataArray(unincluded_array, coords=unincluded_coords)
    
    augmented_da = xr.concat([storm_da, unincluded_da], dim='time')
    augmented_da = augmented_da.rolling(time=8, min_periods=1).max()
    
    return augmented_da

def compute_precip_summaries(storm_da, cell_areas, agg_func):
    
    storm_da = storm_da.assign_coords(lat=storm_da.lat.round(5), lon=storm_da.lon.round(5))
    augmented_da = augment_storm_da(storm_da)
    
    fnames = grab_MERRA2_files(augmented_da, 'tavg1_2d_int_Nx')
    
    var_lst = ['PRECLS', 'PRECCU', 'PRECSN']
    
    ds_lst = []
    for fname in fnames:
        ds = xr.open_dataset(scratch_path + 'merra2_data_precip_ivt/' + fname)
        shifted = ds.assign_coords(time=ds.time - np.timedelta64(30, 'm'))
        upscaled = (shifted[var_lst]*60*60).resample(time='3h').sum()
        ds_lst.append(upscaled)
        
    obs_ds = xr.concat(ds_lst, dim='time')
    obs_ds = obs_ds.assign_coords(lat=obs_ds.lat.round(5), lon=obs_ds.lon.round(5))
    
    summaries = []
    # compute aggregate rainfall
    obs_ds['tot_rainfall'] = obs_ds['PRECCU'] + obs_ds['PRECLS']
    summaries.append(agg_func(augmented_da, obs_ds['tot_rainfall'], cell_areas))
    # compute aggregate snowfall
    summaries.append(agg_func(augmented_da, obs_ds['PRECSN'], cell_areas))
    
    return summaries

##################################### Compute quantities from inst1_2d_asm_Nx #####################################

print('Beginning masking of quantities from inst1_2d_asm_Nx')
ticker = 'inst1_2d_asm_Nx'

func_vars_dict = {('max_T2m_ais', 'T2M'): lambda storm_da, var_da, area_da: compute_max_intensity(storm_da, var_da, area_da, ais_mask),
                  ('avg_IWV_ais', 'TQV'): lambda storm_da, var_da, area_da: compute_average(storm_da, var_da, area_da, ais_mask),
                  ('max_IWV_ais', 'TQV'): lambda storm_da, var_da, area_da: compute_max_intensity(storm_da, var_da, area_da, ais_mask),
                  ('max_ocean_SLP_gradient', 'SLP'): lambda storm_da, var_da, area_da: compute_max_SLPgrad(storm_da, var_da, area_da, ais_mask)}

func_vars_dict_anomaly = {('max_T2M_anomaly_ais', 'T2M'): lambda storm_da, var_da, area_da: compute_max_intensity(storm_da, var_da, area_da, ais_mask),
                          ('max_IWV_anomaly_ais', 'TQV'): lambda storm_da, var_da, area_da: compute_max_intensity(storm_da, var_da, area_da, ais_mask)}

summaries_lst_inst1 = []

for i in tqdm(range(landfalling_storms.shape[0])):
    
    storm = landfalling_storms.iloc[i].data_array
    summaries = compute_raw_summaries(storm, func_vars_dict, cell_areas, ticker, inst1_data_path)
    summaries_anomaly = compute_anomaly_summaries(storm, func_vars_dict_anomaly, climatology_ds, cell_areas, ticker, inst1_data_path)
    
    summaries = summaries + summaries_anomaly
    summaries_lst_inst1.append(summaries)
    
labels_inst1 = np.append(np.array(list(func_vars_dict.keys()))[:,0], np.array(list(func_vars_dict_anomaly.keys()))[:,0])

##################################### Compute quantities from tavg1_2d_asm_Nx #####################################
print('Beginning masking of quantities from tavg1_2d_asm_Nx')

ticker = 'tavg1_2d_int_Nx'
func_vars_dict = {('avg_vIVT_ais', 'VFLXQV'): lambda storm_da, var_da, area_da: compute_average(storm_da, -var_da, area_da, ais_mask),
                  ('max_vIVT_ais', 'VFLXQV'): lambda storm_da, var_da, area_da: compute_max_intensity(storm_da, -var_da, area_da, ais_mask),
                  ('avg_vIVT', 'VFLXQV'): lambda storm_da, var_da, area_da: compute_average(storm_da, -var_da, area_da),
                  ('max_vIVT', 'VFLXQV'): lambda storm_da, var_da, area_da: compute_max_intensity(storm_da, -var_da, area_da)}

summaries_lst_tavg1_precip = []

for i in tqdm(range(landfalling_storms.shape[0])):
    
    storm = landfalling_storms.iloc[i].data_array
    summaries_ivt = compute_raw_summaries(storm, func_vars_dict, cell_areas, ticker, tavg1_precip_data_path, ivt=True)
    summaries_precip = compute_precip_summaries(storm, cell_areas, lambda storm_da, var_da, area_da: compute_cumulative(storm_da, var_da, area_da, ais_mask))
    summaries = summaries_ivt + summaries_precip
    summaries_lst_tavg1_precip.append(summaries)
    
labels_tavg1_precip = np.append(np.array(list(func_vars_dict.keys()))[:,0], np.array(['cumulative_rainfall_ais', 'cumulative_snowfall_ais']))

##################################### Compute quantities from tavg1_2d_slv_Nx #####################################
print('Beginning masking of quantities from tavg1_2d_slv_Nx')

ticker = 'tavg1_2d_slv_Nx'
func_vars_dict = {('max_landfalling_v850hPa', 'V850'): lambda storm_da, var_da, area_da: compute_max_landfalling_wind(storm_da, -var_da, area_da, ais_mask),
                 ('avg_landfalling_v850hPa', 'V850'): lambda storm_da, var_da, area_da: compute_avg_landfalling_wind(storm_da, -var_da, area_da, ais_mask)}

summaries_lst_tavg1_wind = []

for i in tqdm(range(landfalling_storms.shape[0])):
    
    storm = landfalling_storms.iloc[i].data_array
    summaries = compute_raw_summaries(storm, func_vars_dict, cell_areas, ticker, tavg1_850hPa_wind_data_path, ivt=True)
    summaries_lst_tavg1_wind.append(summaries)
    
labels_tavg1_wind = np.array(list(func_vars_dict.keys()))[:,0]

##################################### Compute spatial and durational quantities #####################################

landfalling_storms['max_area'] = landfalling_storms['data_array'].apply(compute_max_area)
landfalling_storms['mean_area'] = landfalling_storms['data_array'].apply(compute_mean_area)
landfalling_storms['mean_landfalling_area'] = landfalling_storms['data_array'].apply(lambda x: compute_mean_area(x, ais_mask))
landfalling_storms['cumulative_landfalling_area'] = landfalling_storms['data_array'].apply(lambda x: compute_cumulative_spacetime(x, ais_mask))
landfalling_storms['duration'] = landfalling_storms['data_array'].apply(compute_duration)
landfalling_storms['start_date'] = landfalling_storms['data_array'].apply(add_start_date)
landfalling_storms['end_date'] = landfalling_storms['data_array'].apply(add_end_date)
landfalling_storms['max_south_extent'] = landfalling_storms['data_array'].apply(compute_max_southward_extent)

# region definitions from Wisconsin folks
region_defs = {'MBL': [-150, -120], 
               'ELS': [-120, -75],
               'AP': [-75, -50],
               'FRIS': [-50, -30],
               'QML': [-30, 30],
               'END': [30, 75],
               'QMC': [75, 120],
               'WLK': [120, 150],
               'VICT': [150, 180],
               'RIS': [-180, -150]}
region_masks = find_region_masks(region_defs, ais_mask)
landfalling_storms['region'] = landfalling_storms['data_array'].apply(lambda x: find_landfalling_region(x, region_masks))
landfalling_storms['trajectory'] = landfalling_storms['data_array'].apply(extract_trajectory)

# add in the merra2 aggregates to the landfalling dataframe
landfalling_storms[labels_inst1] = summaries_lst_inst1
landfalling_storms[labels_tavg1_precip] = summaries_lst_tavg1_precip
landfalling_storms[labels_tavg1_wind] = summaries_lst_tavg1_wind


# save the dataframe
landfalling_storms.to_hdf(home_dir + '/data/ar_database/dataframe_eps12_eps500_minpts5_reppts10/landfalling_storm_quantities_df.h5', key='df')
