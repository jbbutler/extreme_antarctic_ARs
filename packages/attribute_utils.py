'''
Module with functions to compute attributes of storms.

Jimmy Butler
October 2025
'''

from loading_utils import load_ais

ais_mask = load_ais()

def is_landfalling(ar_da):
    '''
    Function to determine if a given AR has made landfall on the AIS.

    Inputs:
        ar_da (xarray.DataArray): binary valued DataArray representing the ARs footprint

    Outputs:
        is_landfalling (boolean): whether the AR intersected the AIS at any point in time
    '''

    is_landfalling = (ais_mask*ar_da).any().values

    return is_landfalling

