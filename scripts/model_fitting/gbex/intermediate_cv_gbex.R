# Intermediate Quantile Regression Model Selection via Hyperparameter Grid Search
# and 5-fold CV
#
# Jimmy Butler
# 11/17/2025

library(tidyverse)
library(argparse)
library(parallel)
library(caret)
library(jsonlite)
library(foreach)
library(doSNOW)
library(here)

root_dir <- here()

source('intermediate_cv_utils.R')

parser <- ArgumentParser(description = 'A script to run a hyperparameter search for modelling intermediate conditional quantiles.')
parser$add_argument('--x_cols', required=TRUE, nargs='+', help='A list of predictor variables.')
parser$add_argument('--y_col', required=TRUE, help='The outcome variable.')
parser$add_argument('--hyperparam_json', required=TRUE, help='A JSON of hyperparameters to use in our grid search.')
parser$add_argument('--chunk_size', required=TRUE, type='integer', help='Number of hyperparam combos to process in parallel.')
parser$add_argument('--ncores', required=TRUE, type='integer', help='The number of cores to parallelize the search over.')
parser$add_argument('--save_name', required=TRUE, help='The file to save hyperparam search results to.')

# for reproducibility in generating folds
set.seed(56789)

args <- parser$parse_args()

train_dat_full <- read_csv(paste0(root_dir, '/project/dataset/datasets/model_ready/train.csv'))
X <- train_dat_full %>% select(args$x_cols)
y <- train_dat_full %>% select(args$y_col) %>% pull()

n_folds <- 5 # number of folds for CV
cv_folds <- createFolds(y, k=n_folds, returnTrain=FALSE)

# load up and process the hyperparam grid JSON
hyperparam_path <- paste0(root_dir,
                          '/project/modelling/gbex/cross_validation/rounds_intermediate/',
                          args$hyperparam_json)
model_params <- fromJSON(hyperparam_path)
param_grid <- expand.grid(model_params[c('num_trees', 'min_node_size', 'sample_frac', 'alpha')])
tau_0 <- model_params$intermediate_quantile

n_chunks <- ceiling(nrow(param_grid)/args$chunk_size)
groups <- gl(n_chunks, args$chunk_size, length=nrow(param_grid))
chunk_lst <- split(param_grid, groups)
n_cores <- args$ncores

# a function that runs the cross validation scheme over a chunk of hyperparameters
# we wish to execute this function over separate chunks of hyperparams in parallel
parallelizedCode <- function(ind) {

    chunk_grid <- chunk_lst[[ind]]
    chunk_results <- list()

    for (i in 1:nrow(chunk_grid)) {
        params <- chunk_grid[i,]
        fold_losses <- c()
  
        for (k in 1:n_folds) {
            # split the data
            test_inds <- cv_folds[[k]]
            train_X <- X[-test_inds,]
            train_y <- y[-test_inds]
            test_X <- X[test_inds,]
            test_y <- y[test_inds]
    
            # Fit the quantile forest
            fit_thresh <- quantile_forest(
                train_X, train_y,
                quantiles = tau_0,
                num.trees = params$num_trees,
                min.node.size = params$min_node_size,
                sample.fraction = params$sample_frac,
                alpha = params$alpha,
                honesty=model_params$honesty,
                seed=12345)
    
            thold_quantiles <- predict(fit_thresh, test_X, quantiles = tau_0)$predictions[, 1]
            loss <- pinball_loss(test_y, thold_quantiles, tau_0)
            fold_losses <- c(fold_losses, loss)
          }
  
        mean_cv_loss <- mean(fold_losses, na.rm = TRUE)
        chunk_results[[i]] <- data.frame(params, mean_cv_loss = mean_cv_loss)
    }
    chunk_results <- do.call(rbind, chunk_results)
    return(chunk_results)
}

# parallel loop execution
clust <- makeSOCKcluster(n_cores)
registerDoSNOW(clust)
pb <- txtProgressBar(min = 1, max = n_chunks, style = 3)
progress <- function(n) setTxtProgressBar(pb ,n)
opts <- list(progress = progress)

cv_grid_results <- foreach(i = 1:length(chunk_lst), 
                .options.snow = opts, 
                .packages = c('grf'), 
                .combine='rbind') %dopar% parallelizedCode(i)

close(pb)
stopCluster(clust)

# save the loss alongside grid
write_csv(cv_grid_results, paste0('rounds_intermediate/', args$save_name))