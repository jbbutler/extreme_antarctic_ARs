'''
A module with functions to load up any datasets that may be useful in other computations,
including the Wille (2024) catalogs, and the AIS points.

Jimmy Butler
October 2025
'''

from pathlib import Path
import xarray as xr
import os
import numpy as np

cur_path = Path(__file__)
home_dir = Path(cur_path).parents[1]

def load_catalogs(years=None):
    '''
    Load up the Wille 2024 catalogs. By default 1980-2022 are loaded up.
        Removes any times for which there are no ARs present, and subsets to
        just the region we are interested in (AIS + Southern Ocean)

    Inputs:
        years (list): the years you would like to load, if not all of them
        
    Outputs:
        catalog_subset (xarray.DataArray): the binary valued DataArray from Wille 2024
    '''

    catalog_paths = str(home_dir/'project/input_data/wille_ar_catalogs/*.nc')
    full_catalog = xr.open_mfdataset(catalog_paths)

    if years is not None:
        full_catalog = full_catalog.sel(time=full_catalog.time.dt.year.isin(years))

    # get rid of all non-antarctic points
    catalog_subset = full_catalog.sel(lat=slice(-86, -39)).ar_binary_tag
    # get rid of all time steps for which there is no AR present
    is_ar_time = catalog_subset.any(dim = ['lat', 'lon'])
    catalog_subset = catalog_subset.sel(time=is_ar_time)

    return catalog_subset

def load_ais(points=False):
    '''
    Load up the AIS mask.

    Inputs:
        points (boolean): if True, gives a list of coordinate cells that correspond to the AIS.
            By default, loads up the binary valued xarray.DataArray mask.
    Outputs:
        Depends on points, as above.
    '''

    # Load up the AIS mask
    mask_path = home_dir/'project/input_data/antarctic_masks/AIS_Full_basins_Zwally_MERRA2grid_new.nc'
    full_ais_mask = xr.open_dataset(mask_path).Zwallybasins > 0
    # grab only points in the Southern Ocean area
    ais_mask = full_ais_mask.sel(lat=slice(-86, -39))

    if points:
        # get ais points
        ais_mask_lats = ais_mask.lat[np.where(ais_mask.to_numpy())[0]].to_numpy()
        ais_mask_lons = ais_mask.lon[np.where(ais_mask.to_numpy())[1]].to_numpy()
        ais_pts = set(zip(ais_mask_lats, ais_mask_lons))

        return ais_pts

    return ais_mask

