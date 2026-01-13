---
title: Quantifying the association between Antarctic AR characteristics and their
  impacts using extreme-value statistics
exports:
  - format: pdf
    output: exports/writeup.pdf
---
+++
Though Antarctic ARs are classically detected based on an extreme threshold of poleward water vapor transport, they differ from each other in characteristics like moisture content, windiness, landfalling orientation to the continent, spatial extent, and landfalling duration. In this study, we quantify the strength of association between such AR characteristics and their extreme thermal and precipitation impacts on the Antarctic Ice Sheet. To accomplish this, we apply density-based clustering algorithms to an existing Eulerian catalog to track Antarctic ARs as discrete objects and construct a database of these events. The database includes summary statistics that characterize each storm and its impacts, such as maximum/cumulative landfalling precipitation and temperature, spatial extent, duration, maximum integrated vapor transport, and landfalling orientation, facilitating investigation of AR lifecyles and their varying characteristics on an Antarctic-wide scale. We then investigate relationships between AR characteristics and impact variables using extreme-value statistical methods. Fill in more details about the methods and findings later..

+++
## Introduction

+++
## Tracking ARs

A commonly used catalog in analyses of Antarctic ARs is that of Wille 2021, which identifies the presence of an AR at a particular pixel in space and time by whether that pixels poleward integrated vapor transport value, or vIVT, exceeds the 98th percentile of vIVT values at that location for that month. While successfully employed in numerous studies, the catalog does not distinguish groups of pixels consisting of one AR event from another, which is needed for our analysis of storm-by-storm comparisons of characteristics and landfalling impacts. Therefore, our first task was to build a catalog of events that distinguishes and tracks separate AR events through space and time.

(section:method)=
### Method

The problem of partitioning groups of AR pixels into individual storm events can be viewed as an unsupervised clustering problem, where there are no training labels present indicating whether certain groups of pixels are part of the same AR. Additionally, the number of clusters, or ARs, is unknown a priori. Given these constraints, our AR identification procedure makes heavy use of the popular `DBSCAN` (Density Based Spatial Clustering of Applications with Noise) algorithm, which seeks to identify clusters in datasets by finding dense groupings of points with respect to a user-specified distance metric, informally defining a cluster as enough points within an {math}`\epsilon`-neighborhood of each other (how much is "enough" is specified by the user), with all other points being relegated to noise.

Since several ARs could be present around Antarctica at the same time, we split this problem into two separate stages:

1.  Spatial clustering of AR pixels within each time step

2.  Spatiotemporal stitching of clusters identified in Step 1

3.  
To accomplish Step 1, we use the `DBSCAN`. Since the `DBSCAN` algorithm in `scikit-learn` does not support temporal distance measures, we use its spatiotemporal extension, the `ST-DBSCAN` algorithm from Birant (2007). We could not find a published software implementation of this algorithm, so we coded it up ourselves. Before detailing our AR-tracking procedure, we first informally review how the methods find clusters and what hyperparameters must be specified.

#### The `DBSCAN` Algorithm

Suppose we have some real-valued distance metric {math}`d({\cdot, \cdot})` that gives us some notion of distance between two points. `DBSCAN` categorizes points as either being core points, border points, or noise. A point {math}`p` is a core point if there is at least {math}`n_{min}` points within an {math}`\epsilon` neighborhood of {math}`p`. The presence of a core point is a defining characteristic of a cluster. Now, if a point {math}`p` is within an {math}`\epsilon` neighborhood of core point {math}`q`, we say {math}`p` is *directly density reachable* from {math}`q`. In this case, {math}`p` itself may still be a core point with many neighbors. If we keep working our way outside the cluster and finding chains of points which are directly density reachable from the last, eventually we will reach a point which itself is not a core point. Call this point {math}`p` a border point, and define this point as being *density-reachable* from {math}`q`. If two border points {math}`p_{1}` and {math}`p_{2}` are actually part of the same cluster, then those points must be *density-connected*, meaning that there exists a core point {math}`q` within the cluster to which {math}`p_{1}` and {math}`p_{2}` are each *density-reachable*. More generally, a cluster is defined as all those points which are density-reachable from known cluster points, and are density connected to each other. All other unclassified points receive the noise label. The algorithm works by going through each point individually; if we are at a core point, the algorithm successively visits all neighbors until border points are reached, and if we are not at a core point, the algorithm skips to the next point. Any true border points will be given the requisite label, while noise points will remain unlabelled after all points have been visited.

From this description, we see three primary hyperparameters: {math}`d(\cdot, \cdot)` (the distance metric), {math}`n_{min}` (the minimum number of neighbors required to be a core point), and {math}`\epsilon` (the neighborhood size). One can imagine how altering these parameters impacts the number of clusters observed: a smaller {math}`\epsilon` leads to less points being within neighbors of each other, leading to more clusters detected, while a larger {math}`n_{min}` leads to less points being categorized as core points and thus less clusters and more noise.

#### The `ST-DBSCAN` Algorithm

Our problem is inherently a spatiotemporal clustering problem, where notions of distance between two points must incorporate some aspect of their spatial distance as well as distance in time. In the `scikit-learn` implementation of the above algorithm, no available distance metric incorporates both time and space. So, we use the `ST-DBSCAN` algorithm to handle any spatiotemporal clustering. The algorithm is almost identical to the usual `DBSCAN` algorithm, except our notion of two points being neighbors requires conditions on a spatial distance metric {math}`d_{space}(\cdot, \cdot)` and a time distance metric {math}`d_{time}(\cdot, \cdot)`. A point {math}`p` is thus within a neighborhood of {math}`q` is {math}`d_{time}(p,q) < \epsilon_{time}` and {math}`d_{space}(p,q) < \epsilon_{space}`. The rest of the algorithm then proceeds as in the usual case. Note this adds an extra hyperparameter to consider.

With these definitions and procedures laid out, we track ARs in the following way. First, consider the {math}`i`th AR pixel in the dataset as a three-vector {math}`p_{i} = (\theta_{i}, \phi_{i}, t_{i})`, where {math}`\theta_{i}` and {math}`\phi_{i}` are the latitude and longitude coordinates in radians, respectively, and {math}`t_{i}` is the time at which that pixel was associated with AR conditions. For any two points {math}`p_{i}, p_{j}`, define their spatial distance {math}`d_{space}(p_{i}, p_{j})` as the Haversine distance, and define their temporal distance {math}`d_{time}(p_{i}, p_{j})` as the number of hours separating the two points. Choose the spatial neighborhood size {math}`\epsilon_{space}=500` km, or half the synoptic scale, and the time neighborhood size {math}`\epsilon_{time}=12` hours. Further, we choose {math}`n_{min} = 5`. For each time step in which at least one AR pixel is present, we use DBSCAN with {math}`d_{space}`, {math}`\epsilon_{space}`, and {math}`n_{min}` to group the pixels into potentially several AR events. From each identified cluster, we randomly sample {math}`N` many points without replacement, which we will consider "representative" points of that cluster (if the cluster contains less than {math}`N = 10` points, the all points in the cluster will be considered representative points). Figure [1](#fig:spatial_clustering) visualizes this process for a particular AR timestep. After running through each AR time step accordingly, we run the ST-DBSCAN algorithm on all of the representative points across all years, clustering them spatiotemporally. In the rare case where some of the representative points within the same previously-identified spatial cluster are given different labels, we assign the label which was most commonly given. This procedure is visualized in Figure [2](#fig:spatiotemporal_clustering). All of the original points in each spatial cluster identified in the first stage are then given this label.

:::{figure} paper_plots/spatial_clustering.png
:name: fig:spatial_clustering
:align: center

(a) For identified AR pixels on February 8, 1980 at 06:00 UTC in the Wille 2021 catalog, (b) the DBSCAN algorithm partitions the AR pixels into separate events, and then (c) {math}`N=10` representative points are sampled from each identified cluster (c).
:::

:::{figure} paper_plots/spatiotemporal_clustering.png
:name: fig:spatiotemporal_clustering
:align: center

(a) After spatially clustering and randomly sampling AR pixels from each cluster on February 8, 1980 at each of 00:00, 06:00, 12:00, and 18:00 UTC, (b) ST-DBSCAN is used to stitch the clusters together, effectively tracking the identified ARs over time with a single label.
:::

These hyperparameters were chosen on the basis of the physicality and typical behavior of ARs: if AR conditions at a location in space appear outside of a 12 hour window, then this is likely a different AR event, and since ARs are synoptic-scale phenomena but smaller than their mid-latitude counterparts due to the Coriolis effect, any AR more than 500 km away from another can be considered to be a separate storm event. Through careful examination of algorithmic output and comparisons to existing case studies, we find these sets of hyperparameters faithfully identify and track AR events in the Wille 2021 catalog. See the supplementary material for examples of case-studies from the AR literature that have been successfully identified, as well as a sensitivity analysis exploring how the total number of ARs detected are affected by perturbations of the hyperparameters.

### AR Catalog Product

We now take a moment to describe the structure of our tracked AR catalog from a software perspective, and how it compares to the original Wille 2021 threshold catalog. As shown in Figure [3](#fig:catalog_tracker_comparison), the original Wille 2021 catalog consists of a binary-valued `xarray.DataArray`, indicating whether a pixel at a particular time is experiencing AR conditions. Thus, this effectively is a binary mask for all AR pixels throughout the given time period. After extracting these AR pixels and running the clustering procedure described in Section [??](#section:method), the pixels associated with each identified cluster, or storm, are gathered into binary masks (in the form of `xarray.DataArray` objects). The dimensions of each binary mask depend on the max/min latitudinal/longitudinal extents of the storm throughout its lifetime, and for how many time steps the cluster existed. Thus, each binary mask is the smallest binary-valued `xarray.DataArray` cube that can contain all of the pixels associated with that storm. These `xarray.DataArray` objects are then arranged as a column in a `pandas.DataFrame` object, where each row corresponds to a different AR.

:::{figure} paper_plots/catalog_tracker_comparison.png
:name: fig:catalog_tracker_comparison
:align: center

The poleward vIVT catalog is structured as an `xarray.DataArray` object that masks all of the identified AR pixels (left), while our tracked catalog is structured as a `pandas.DataFrame` where each row contains a binary-valued mask of just that storm’s pixels.
:::

We choose to structure our catalog as a `pandas.DataFrame` to take advantage of the package’s powerful API to extract variable information from each AR with minimal code. For instance, suppose one wishes to add a column to the DataFrame indicating whether each AR made landfall over the Antarctic Ice Sheet. Assuming one can write a simple function to determine if a single AR made landfall (using the storm mask and a mask for the AIS), then adding the column is a simple one line of code with a call to `pandas.Series.apply()`. It’s also simple to extract the start and end date of each storm from their binary masks: it’s simply just the first and last timestamps on that `xarray.DataArray` object’s `time` dimension. To extract atmospheric quantities associated with each AR, using a reanalysis product of interest (we opt for MERRA-2 as we use this version of the Wille 2021 catalog), one can simply mask into the `xarray.DataArray` for that variable and obtain scalar summaries for each desired quantity. This includes quantities like cumulative snowfall on the AIS, maxmimum sea-level pressure gradient underneath the AR footprint, and many more. More detailed information on how to leverage this catalog’s structure to extract these, and other, quantities (including code) is contained on [the catalog documentation site](jbbutler.github.io/ar-catalog-demo/), underneath the ’How to use this catalog’ tab.

### AR Catalog Summaries

Before extracting atmospheric quantities from reanalysis datasets, there is plenty of information that can be gleaned about ARs only using their binary masks. In this section, we focus on key findings that both are easily computed and summarize the overall occurrence of ARs, including quantities related to AR frequency and landfalling geographic region. For details on how the AR catalog was cleaned and preprocessed, see Supplementary Material A.

+++
## Characteristics and Impacts of ARs

+++
## Quantifying Associations

### Method

### Results

+++
## Discussion

+++
## Conclusions and Future Work