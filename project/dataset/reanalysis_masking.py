'''
Script that will fill in the dataframe of AR events with relevant information 
about each storm, like durations, landfalling regions as well as masking re-analysis 
data to obtain quantities like snowfall, moisture content, energy, etc.

Jimmy Butler
June 2025
'''
import pandas as pd
import xarray as xr
import numpy as np
import os
from pathlib import Path
import xarray as xr
from tqdm import tqdm
import sys

curwd = os.getcwd()
home_dir = Path(curwd).parents[1]
sys.path.append(str(home_dir/'packages/'))
from st_dbscan import utils
import loading_utils
from attribute_utils import *
from compute_attributes import *

# configure paths to where each MERRA-2 dataset data is stored in my Perlmutter scratch folders
# ideally in the future, this could just be replaced with masking data streamed directly
scratch_path = '/pscratch/sd/j/jbbutler/'
inst1_data_path = '/pscratch/sd/j/jbbutler/merra2_data_T2m_V10m_SLP_IWV/'
tavg1_precip_data_path = '/pscratch/sd/j/jbbutler/merra2_data_precip_ivt/'
tavg1_850hPa_wind_data_path = '/pscratch/sd/j/jbbutler/merra2_data_850hPa_wind/'
inst3_omega_data_path = '/pscratch/sd/j/jbbutler/merra2_data_omega/'

# loading a bunch of different datasets for computation
df_path = home_dir/'project/catalog/epsspace0.5_epstime12_minpts5_nreppts10_seed12345.h5'
dataframe = pd.read_hdf(df_path)
landfalling_storms = dataframe[dataframe.is_landfalling]

cell_areas = loading_utils.load_cell_areas()
cell_areas = cell_areas.assign_coords(lat=cell_areas.lat.round(5), 
                                      lon=cell_areas.lon.round(5)) # this is to avoid -0 not matching 0

ais_mask = loading_utils.load_ais()
ais_mask = ais_mask.assign_coords(lat=ais_mask.lat.round(5),
                                  lon=ais_mask.lon.round(5))

elevation = loading_utils.load_elevation()

# compute the climatologies (so far, only for SLP, 2m-temperature, and IWV/TQV)
# to be used for SLP, 2m-temp, and IWV anomalies
monthly_averages = xr.open_mfdataset('/pscratch/sd/j/jbbutler/merra2_monthly_data/*.nc4')
climatology_ds = monthly_averages.groupby(monthly_averages.time.dt.month).mean().compute()

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
    summaries_ivt = compute_raw_summaries(storm, func_vars_dict, cell_areas, ticker, tavg1_precip_data_path, half_hour=True)
    summaries_precip = compute_precip_summaries(storm, cell_areas, lambda storm_da, var_da, area_da: compute_cumulative(storm_da, var_da, area_da, ais_mask), tavg1_precip_data_path)
    summaries = summaries_ivt + summaries_precip
    summaries_lst_tavg1_precip.append(summaries)
    
labels_tavg1_precip = np.append(np.array(list(func_vars_dict.keys()))[:,0], np.array(['cumulative_rainfall_ais', 'cumulative_snowfall_ais']))

##################################### Compute wind quantities from tavg1_2d_slv_Nx #####################################
print('Beginning masking of wind quantities from tavg1_2d_slv_Nx')

ticker = 'tavg1_2d_slv_Nx'
func_vars_dict = {('max_landfalling_v850hPa', 'V850'): lambda storm_da, var_da, area_da: compute_max_landfalling_wind(storm_da, -var_da, area_da, ais_mask),
                 ('avg_landfalling_v850hPa', 'V850'): lambda storm_da, var_da, area_da: compute_avg_landfalling_wind(storm_da, -var_da, area_da, ais_mask)}

summaries_lst_tavg1_wind = []

for i in tqdm(range(landfalling_storms.shape[0])):
    
    storm = landfalling_storms.iloc[i].data_array
    summaries = compute_raw_summaries(storm, func_vars_dict, cell_areas, ticker, tavg1_850hPa_wind_data_path, half_hour=True)
    summaries_lst_tavg1_wind.append(summaries)
    
labels_tavg1_wind = np.array(list(func_vars_dict.keys()))[:,0]

##################################### Compute omega quantities from inst3_3d_asm_Np #####################################

print('Beginning masking of omega from inst3_3d_asm_Np')

ticker = 'inst3_3d_asm_Np'

func_vars_dict = {('avg_landfalling_minomega', 'OMEGA'): lambda storm_da, var_da, area_da: compute_avg_landfalling_minomega(storm_da, var_da, area_da, ais_mask)}

summaries_lst_inst3_omega = []

for i in tqdm(range(landfalling_storms.shape[0])):
    
    storm = landfalling_storms.iloc[i].data_array
    summaries = compute_raw_summaries(storm, func_vars_dict, cell_areas, ticker, inst3_omega_data_path)
    summaries_lst_inst3_omega.append(summaries)
    
labels_inst3_omega = np.array(list(func_vars_dict.keys()))[:,0]

##################################### Compute spatial and durational quantities #####################################

landfalling_storms['max_area'] = landfalling_storms['data_array'].apply(compute_max_area)
landfalling_storms['mean_area'] = landfalling_storms['data_array'].apply(compute_mean_area)
landfalling_storms['mean_landfalling_area'] = landfalling_storms['data_array'].apply(lambda x: compute_mean_area(x, ais_mask))
landfalling_storms['cumulative_landfalling_area'] = landfalling_storms['data_array'].apply(lambda x: compute_cumulative_spacetime(x, ais_mask))
landfalling_storms['duration'] = landfalling_storms['data_array'].apply(compute_duration)
landfalling_storms['start_date'] = landfalling_storms['data_array'].apply(add_start_date)
landfalling_storms['end_date'] = landfalling_storms['data_array'].apply(add_end_date)
landfalling_storms['max_south_extent'] = landfalling_storms['data_array'].apply(compute_max_southward_extent)
landfalling_storms['max_elevation_grad'] = landfalling_storms['data_array'].apply(lambda x: compute_max_elevation_grad(x, elevation))

region_defs_coarser = {'West': [-150, -30], 
               'East 1': [-30, 75],
               'East 2': [75, -150]}

region_masks_coarser = find_region_masks(region_defs_coarser, ais_mask)

landfalling_storms['coarser_region'] = landfalling_storms['data_array'].apply(lambda x: find_landfalling_region(x, region_masks_coarser))

landfalling_storms['trajectory'] = landfalling_storms['data_array'].apply(extract_trajectory)

# add in the merra2 aggregates to the landfalling dataframe
landfalling_storms[labels_inst1] = summaries_lst_inst1
landfalling_storms[labels_tavg1_precip] = summaries_lst_tavg1_precip
landfalling_storms[labels_tavg1_wind] = summaries_lst_tavg1_wind
landfalling_storms[labels_inst3_omega] = summaries_lst_inst3_omega


# save the dataframe
landfalling_storms.to_hdf(home_dir/'project/dataset/datasets/landfalling_storm_quantities_df.h5', key='df')
