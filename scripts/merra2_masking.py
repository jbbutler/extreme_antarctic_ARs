import pandas as pd
import xarray as xr
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import xarray as xr
from tqdm import tqdm

import boto3
import s3fs
import earthaccess
from IPython.display import display, Markdown
from ipywidgets import IntProgress

home_dir = str(Path(os.getcwd()).parents[0])

# load up all of the dataframes by year, and then concatenate into one big one
df_path = home_dir + '/data/ar_database/dataframes/'
fnames = os.listdir(df_path)
df_list = []

for fname in fnames:
    df_list.append(pd.read_pickle(df_path + fname))
    
dataframe = pd.concat(df_list)

cell_areas = xr.open_dataset('/home/jovyan/extreme_antarctic_ARs/data/area/MERRA2_gridarea.nc')
cell_areas = cell_areas.cell_area
ais_mask = xr.open_dataset('/home/jovyan/extreme_antarctic_ARs/data/antarctic_masks/AIS_Full_basins_Zwally_MERRA2grid_new.nc')
ais_mask = ais_mask > 0



def compute_cumulative(storm_da, var_da, area_da, ais_da=None):
    if ais_da:
        storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
        storm_da_subset = storm_da.where(storm_ais_mask, 0)
    else:
        storm_da_subset = storm_da.copy()

    var_da_subset = var_da.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)

    storm_cell_areas = area_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    avg_storm_val = storm_cell_areas.dot(storm_da_subset*var_da_subset).mean().values

    return avg_storm_val

def compute_max_intensity(storm_da, var_da, ais_da=None):
    if ais_da:
        storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
        storm_da_subset = storm_da.where(storm_ais_mask, 0)
    else:
        storm_da_subset = storm_da.copy()
        
    var_da_subset = var_da.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)
    max_intensity_val = (storm_da_subset*var_da_subset).max().values

    return max_intensity_val

def compute_average(storm_da, var_da, area_da, ais_da=None):
    if ais_da:
        storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon).Zwallybasins
        storm_da_subset = storm_da.where(storm_ais_mask, 0)
    else:
        storm_da_subset = storm_da.copy()

    var_da_subset = var_da.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)

    storm_cell_areas = area_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    tot_area = storm_da_subset.dot(storm_cell_areas)
    avg_storm_val = (storm_cell_areas.dot(storm_da_subset*var_da_subset)/tot_area).mean().values

    return avg_storm_val

auth = earthaccess.login()

data_doi = '10.5067/3Z173KIE2TPD'
vars = ['T2M']

dataframe_landfalling = dataframe[dataframe.is_landfalling]
t2m_vals = [None]*len(dataframe_landfalling)

for i in tqdm(range(len(dataframe_landfalling))):

    storm_da = dataframe.iloc[i].data_array
    storm_da = storm_da.rename({'lats':'lat', 'lons':'lon'})

    first = np.min(storm_da.time.dt.date.to_numpy())
    last = np.max(storm_da.time.dt.date.to_numpy())
    # stream the data only between those two dates
    results = earthaccess.search_data(doi=data_doi, 
                                  temporal=(f'{first.year}-{first.month}-{first.day}', 
                                            f'{last.year}-{last.month}-{last.day}'));
    obs_ds = xr.open_mfdataset(earthaccess.open(results));
    obs_ds = obs_ds.sel(time = obs_ds.time.dt.hour % 3 == 0).sel(lat = slice(-86, -39))
    obs_ds = obs_ds[vars]

    t2m_vals[i] = compute_average(storm_da, obs_ds.T2M, cell_areas, ais_mask)