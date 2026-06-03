# Linking Antarctic Atmospheric River Characteristics with their Landfalling Impacts: An Exploratory Analysis

A project to investigate the associations between Antarctic atmospheric river characteristics and impacts on the Antarctic ice sheet.

To reproduce the results of this work, you must first create this project's `conda` environment to access the relevant packages. There are two environments relevant for this work, one with `R` packages and one with `python` packages.

To create the `python` environment, run the following in your terminal:

```
conda env create -f environment_python.yml
conda activate extreme_antarctic_ars_python
```

This conda environment, among other packages, also installs `artools`, a package to facilitate the construction Antarctic AR datasets as presented in the paper *Constructing Event-Based Datasets of Geophysical Phenomena from Gridded Products: An Application to Antarctic Atmospheric Rivers*.

To access this environment in the Jupyter Notebooks, run the following line as well:

```
python -m ipykernel install --user --name extreme_antarctic_ars
```

To create the `R` environment, run the following in your terminal:

```
conda env create -f environment_R.yml
conda activate extreme_antarctic_ars_R
```

This environment is primarily used to interface with the [Gradient Boosting for Extremes](https://doi.org/10.1007/s10687-023-00473-x) software, which is currently only implemented in R.

To access this environment in the relevant notebook, first start an R session:

```
R
```

and in that session, run:

```
IRkernel::installspec(name = 'extreme_antarctic_ars_R', displayname = 'extreme_antarctic_ars_R', user = TRUE)
```

You may have to restart your server to see the environment in the kernel dropdown menu.