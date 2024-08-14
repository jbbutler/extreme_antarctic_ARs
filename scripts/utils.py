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