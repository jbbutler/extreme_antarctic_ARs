# Script to do a hyperparameter grid search for the gradient boosted extremes
# model from Velthoen
#
# Jimmy Butler
# 11/2025

library(grf)
library(tidyverse)
library(argparse)
library(parallel)
library(caret)
library(jsonlite)
library(foreach)
library(doSNOW)
library(here)

root_dir <- here()

pacman::p_load_gh('JVelthoen/gbex')

# for reproducibility in generating folds
set.seed(56789)
parser <- ArgumentParser(description = 'A script to run a hyperparameter search for modelling extreme conditional quantiles.')
parser$add_argument('--x_cols', required=TRUE, nargs='+', help='A list of predictor variables.')
parser$add_argument('--y_col', required=TRUE, help='The outcome variable.')
parser$add_argument('--hyperparam_json', required=TRUE, help='A JSON of hyperparameters to use in our grid search.')
parser$add_argument('--grf_hyperparam_json', required=TRUE, help='A set of the best hyperparams to fit intermediate GRF to.')
parser$add_argument('--chunk_size', required=TRUE, type='integer', help='Number of hyperparam combos to process in parallel.')
parser$add_argument('--ncores', required=TRUE, type='integer', help='The number of cores to parallelize the search over.')
parser$add_argument('--save_name', required=TRUE, help='The file to save hyperparam search results to.')

args <- parser$parse_args()

# first, grab training data and fit intermediate threshold model
train_dat_full <- read_csv(paste0(root_dir, '/project/dataset/datasets/model_ready/train.csv'))
X <- train_dat_full %>% select(args$x_cols)
y <- train_dat_full %>% select(args$y_col) %>% pull()

tau_0 <- 0.7 #lower from 0.8 because otherwise only 60 exceedances!
grf_param_path <- paste0(root_dir, '/project/modelling/gbex/cross_validation/', args$grf_hyperparam_json)
best_grf_params <- fromJSON(grf_param_path)

# fit the best grf model
fit_threshold <- quantile_forest(
  X, y,
  quantiles = tau_0,
  num.trees = best_grf_params$num_trees,
  min.node.size = best_grf_params$min_node_size,
  sample.fraction = best_grf_params$sample_frac,
  alpha = best_grf_params$alpha,
  honesty = FALSE,
  seed = 54321
)

# grab the exceedances
u <- predict(fit_threshold, X, quantiles = tau_0)$predictions[,1]
diffs <- y - u
z <- diffs[diffs > 0] # only take exceedances above the intermediate threshold
X <- X[diffs > 0, ] # take the corresponding covariates as well

# load up and process the hyperparam grid JSON
hyperparam_path <- paste0(root_dir,
                          '/project/modelling/gbex/cross_validation/rounds_extreme/',
                          args$hyperparam_json)
model_params <- fromJSON(hyperparam_path)
max_trees <- model_params$max_trees
grid_params <- model_params[names(model_params) != 'max_trees']
gpd_param_grid <- expand.grid(grid_params)

# get chunks over which we will parallelize
n_chunks <- ceiling(nrow(gpd_param_grid)/args$chunk_size)
groups <- gl(n_chunks, args$chunk_size, length=nrow(gpd_param_grid))
chunk_lst <- split(gpd_param_grid, groups)

gpd_grid_search_results <- list()
ncores <- args$ncores

parallelizedCode <- function(ind) {

    chunk_grid <- chunk_lst[[ind]]
    chunk_results <- list()

    for (i in 1:nrow(chunk_grid)) {
        params <- chunk_grid[i, ]
        cv_param_results <- CV_gbex(y=z, X=X, num_folds=5, repeat_cv=1, Bmax=max_trees,
        stratified=FALSE, ncores = 1, depth = c(params$depth_sigma, params$depth_gamma),
        lambda_scale=params$lambda_scale, lambda_ratio=params$lambda_ratio,
        min_leaf_size=c(params$min_leaf_sigma, params$min_leaf_gamma),
        sf=params$sample_frac, silent = TRUE)
  
        avg_val_losses <- cv_param_results$dev_all

        chunk_results[[i]] <- data.frame(
            depth_sigma=params$depth_sigma,
            depth_gamma=params$depth_gamma,
            lambda_scale=params$lambda_scale,
            lambda_ratio=params$lambda_ratio,
            min_leaf_sigma=params$min_leaf_sigma,
            min_leaf_gamma=params$min_leaf_gamma,
            sf=params$sample_frac,
            mean_cv_loss=avg_val_losses,
            num_trees=cv_param_results$par_grid)
    }
    chunk_results <- do.call(rbind, chunk_results)
    return(chunk_results)
}

# parallel loop execution
clust <- makeSOCKcluster(ncores)
registerDoSNOW(clust)
pb <- txtProgressBar(min = 1, max = n_chunks, style = 3)
progress <- function(n) setTxtProgressBar(pb ,n)
opts <- list(progress = progress)

cv_gpd_results <- foreach(i = 1:length(chunk_lst), 
                .options.snow = opts,
                .packages = c('gbex', 'POT', 'treeClust'),
                .combine='rbind') %dopar% parallelizedCode(i)

close(pb)
stopCluster(clust)

# save the loss alongside grid
write_csv(cv_gpd_results, paste0('rounds_extreme/', args$save_name))