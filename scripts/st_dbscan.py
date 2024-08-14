# Implementation of ST_DBSCAN, a version of DBSCAN suited to spatiotemporal clustering problems
# Uses DBSCAN to spatially cluster and reduce data size, then ST-DBSCAN to cluster reduced dataset
# Originally found in "ST-DBSCAN: An algorithm for clustering spatial-temporal data," Birant and Kut, 2007
#
# Code by James Butler
# Date: June 2024

import numpy as np
import pandas as pd
import utils

class ST_DBSCAN:
    
    def __init__(self, eps_space_1, eps_space_2, eps_time, minpts_1, minpts_2, n_rep_pts):

        self.eps_space_1 = eps_space_1
        self.eps_space_2 = eps_space_2
        self.eps_time = eps_time
        self.minpts_1 = minpts_1
        self.minpts_2 = minpts_2
        self.n_rep_pts = n_rep_pts
    
    def fit_spatial(self, data):

        clustering = DBSCAN(eps=eps, min_samples=self.minpts, metric='haversine').fit_predict(X=data)
        fixed_time_df = pd.DataFrame({'lats': data[:,0], 'lons': data[:,1], 'cluster': clustering})

        # get the average lat-lon of each cluster, according to average_angle function above
        avg_positions = pd.DataFrame(fixed_time_df.groupby('cluster').apply(utils.average_angle, include_groups=False).to_list(), 
                        columns=['cluster', 'mean_lat', 'mean_lon'])

        # randomly sample n_rep_pts-many points (without replacement) from each cluster and store
        rep_pts = fixed_time_df.groupby('cluster', as_index=False)[['lats', 'lons']].agg(lambda x: list(np.random.choice(x, min(self.n_rep_pts, len(x)), replace=False)))
        rep_pts.rename(columns={'lats':'rep_lats', 'lons':'rep_lons'}, inplace=True)

        # match avg pos and rep pts by cluster
        rep_pts_df = pd.merge(rep_pts, avg_positions, on='cluster')
        # aggregate ALL lats and lons for each cluster into lists as well
        cluster_info = fixed_time_df.groupby('cluster', as_index=False)[['lats', 'lons']].agg(list)

        # combine all this info into single dataframe
        cluster_info = pd.merge(cluster_info, rep_pts_df, on='cluster')
        # add time into column
        cluster_info['time'] = times[i]
    



