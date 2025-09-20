import os
from pathlib import Path
from tqdm import tqdm
import dask.multiprocessing

import st_dbscan as st

from utils import arctan
from utils import average_angle
from utils import retrieve_neighbors
from utils import construct_da
from utils import is_landfalling
import utils

######## SETUP ########
save_dir = '/scratch/users/butlerj/extreme_antarctic_ars/sensitivity_analysis/all_years/'
catalog_subset = utils.load_catalogs(years=1980)
ais_pts = utils.load_ais()

####### HYPERPARAM VARIATIONS #######
baseline_par_dict = {'seed':12345,
                     'eps_time':12,
                     'eps_space': 0.5,
                     'rep_pts': 10,
                     'min_pts': 5}

#par_perturbations_dict = {'eps_time': [14, 16, 18, 22, 24],
#                         'rep_pts': [10, 20, 30]}

par_perturbations_dict = {'eps_time': [12, 12, 12]}

# make the combos of parameters
combos = []
combos.append(baseline_par_dict)

for key in par_perturbations_dict.keys():
    n_combos = len(par_perturbations_dict[key])
    for i in range(n_combos):
        temp_dict = baseline_par_dict.copy()
        temp_dict[key] = par_perturbations_dict[key][i]
        combos.append(temp_dict)
        
synoptic_scale = 10**3
km_per_radian = 6.371*(10**3)

# set up parallelization using dask
n_tasks = len(combos)
dask.config.set(schedule='processes', num_workers=n_tasks);

# function that executes each iteration of the parallel loop
def parallelized_code(i):
    par_dict = combos[i]

    random.seed(par_dict['seed'])
    
    # make the name for folder to store results in
    fname_tuples = list(par_dict.items())
    name = ''
    for tup in fname_tuples:
        name = name + str(tup[0]) + '_' + str(tup[1])

    name_dir = save_dir + name + '/'    
    #Path(name_dir).mkdir(parents=True, exist_ok=True)
        
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
    dataframe.to_hdf(name_dir + 'storm_df.h5', key='df')
    coord_dict = {'lats': catalog_subset.lat, 'lons': catalog_subset.lon}
    da = utils.construct_dataarray(coord_dict=coord_dict, big_df=obj_subset)
    da.to_netcdf(name_dir + '/storm_da.nc')

future_calls = [dask.delayed(parallelized_code)(i) for i in range(n_tasks)]
dask.compute(future_calls);
