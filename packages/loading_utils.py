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
import earthaccess
import ray

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

def load_cell_areas():
    '''
    Load up the xarray.DataArray with the grid cell areas.

    Inputs:
        None
    Outputs:
        cell_areas (xarray.DataArray): the DataArray in our region of interest with the area of
            each grid cell provided
    '''

    areas_path = home_dir/'project/input_data/area/MERRA2_gridarea.nc'
    cell_areas = xr.open_dataset(areas_path)
    cell_areas = cell_areas.cell_area

    return cell_areas

def load_elevation():
    '''
    Load up the xarray.DataArray with elevation at each grid cell.

    Inputs:
        None
    Outputs:
        elevations (xarray.DataArray): the DataArray in our region of interest with elevations at each grid cell.
    '''

    elevations_path = home_dir/'project/input_data/elevation/Elevation_MERRA2.nc'
    elevations = xr.open_dataset(elevations_path)
    elevations = elevations.PHIS

    return elevations

def grab_MERRA2_files(storm_da, ticker):
    '''
    Grab a list of the MERRA-2 files needed to mask a particular storm.

    Inputs:
        storm_da (xarray.DataArray): the AR's binary mask
        ticker (string): the desired dataset's ID

    Outputs:
        fnames (list): list of the MERRA-2 file names
    '''
    
    dates = np.unique(storm_da.time.dt.date.values)

    fnames = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        fname = ticker + '.' + date_str + '.nc4.nc4'
        fnames.append(fname)

    return fnames

def grab_MERRA2_granules(storm_da, data_doi):
    '''
    Grab a list of data granules from a specific MERRA-2 dataset for an AR,
        specifically pointers to granules stored in Amazon S3 bucket.

    Inputs:
        storm_da (xarray.DataArray) the AR's binary mask
        data_doi (str): the doi of the MERRA-2 dataset

    Outputs:
        list of granule pointers
    '''
    first = np.min(storm_da.time.dt.date.to_numpy())
    last = np.max(storm_da.time.dt.date.to_numpy())
    # stream the data only between those two dates
    granule_lst = earthaccess.search_data(doi=data_doi, 
                                  temporal=(f'{first.year}-{first.month}-{first.day}', 
                                            f'{last.year}-{last.month}-{last.day}'))

    return granule_lst

@ray.remote
class EarthdataGatekeeper:
    '''
    A Ray Actor that makes the open requests to NASA's servers sequentially so that we don't get
        rate limited by NASA.
    '''
    def __init__(self):
        self.auth = earthaccess.login()
    
    def get_granule_pointers(self, granule_lst):
        return earthaccess.open(granule_lst, show_progress=False)