'''
Helper functions to interpret xgboost modelling results.

Jimmy Butler
November 2025
'''

import numpy as np
import pandas as pd

def compute_pdp_1d(feature, X_data, prediction_rule, extents=None, grid_resolution=50):
    '''
    Computes a 1D partial dependence plot on a given feature. Note: we define
        a separate function to do this by hand (as opposed to using sklearn)
        because we use a custom prediction algorithm that implements shrinkage.

    Inputs:
        feature (str): one of the columns in the training dataset
        X_data (pd.DataFrame): dataset to use for computing the pdp
        prediction_rule (function): a function that takes in a single argument,
            a pd.DataFrame of predictors for each observation, and returns
            the predictions
        extents (list): a list of lower and upper extents for the grid. Default: min and max
            of feature in dataset
        grid_resolution (int): how many points to evaluate the partial dependence at

    Outputs:
        grid_values (np.array): the values at which the partial dependence is evaluated
        pdp_values (list): list of partial dependence values evaluated at each point grid_values
    '''

    if extents:
        x_min = extents[0]
        x_max = extents[1]
    else:
        x_min = X_data[feature].min()
        x_max = X_data[feature].max()
        
    grid_values = np.linspace(x_min, x_max, grid_resolution)
    
    pdp_values = []
    
    for val in grid_values:
        X_temp = X_data.copy()
        X_temp[feature] = val
        
        predictions = prediction_rule(X_temp)
        pdp_values.append(np.mean(predictions))
        
    return grid_values, pdp_values

def compute_pdp_2d(feature1, feature2, X_data, prediction_rule, extents=None, grid_resolution=20):
    '''
    Computes a 2D grid of partial dependence values on two provided features.

    Inputs:
        feature1, feature2 (str): the features to vary over the grid
        X_data (pd.DataFrame): dataset to use for computing the pdp
        prediction_rule (function): a function that takes in a single argument,
            a pd.DataFrame of predictors for each observation, and returns
            the predictions
        extents (list):  a list of 4 numbers: feature 1 lower and upper bounds,
            and feature 2 lower and upper bounds (in that order). Default: min and
            max along each axis.
        grid_resolution (int): number of grid points both axes

    Outputs:
        grid1, grid2 (np.array): ticks along each feature axis
        pdp_matrix (np.array): the partial dependence values computed at each grid point
    '''

    if extents:
        grid1 = np.linspace(extents[0], extents[1], grid_resolution)
        grid2 = np.linspace(extents[2], extents[3], grid_resolution)

    else:
        grid1 = np.linspace(X_data[feature1].min(), X_data[feature1].max(), grid_resolution)
        grid2 = np.linspace(X_data[feature2].min(), X_data[feature2].max(), grid_resolution)
    
    pdp_matrix = np.zeros((grid_resolution, grid_resolution))
    
    for i, val1 in enumerate(grid1):
        for j, val2 in enumerate(grid2):
            X_temp = X_data.copy()
            X_temp[feature1] = val1
            X_temp[feature2] = val2
            
            pdp_matrix[j, i] = np.mean(prediction_rule(X_temp))
            
    return grid1, grid2, pdp_matrix

def get_permutation_importance(predict_func, X, y, metric_func, n_repeats=5, maximize=False):
    '''
    Given a prediction rule, compute the importance of a feature by permuting the feature
        column repeatedly and computing the increase in test set error.

    Inputs:
        predict_func (function): the prediction rule that takes in a pd.DataFrame of
            features and returns the outcome
        X (pd.DataFrame): the pd.DataFrame of features
        y (pd.Series): a series of the outcomes corresponding to rows in X
        metric_func (function): function to evaluate goodness (or badness) of predictions
        n_repeats (int): the number of times to permute a particular feature
        maximize (boolean): whether it is desirable to maximize the metric_func
            (example: R2 we want to maximize)
    '''

    baseline_pred = predict_func(X)
    baseline_score = metric_func(baseline_pred, y)

    importances_mean = {}
    importances_std = {}

    for feature in X.columns:
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()     
            X_permuted[feature] = np.random.permutation(X_permuted[feature])      
            perm_pred = predict_func(X_permuted)
            perm_score = metric_func(perm_pred, y)
            scores.append(perm_score)

        avg_perm_score = np.mean(scores)
        
        if maximize:
            importance_val = (baseline_score - avg_perm_score)
        else:
            importance_val = (avg_perm_score - baseline_score)
            
        importances_mean[feature] = importance_val
    results_df = pd.DataFrame({
        'avg_importance': importances_mean
    })
    results_df['avg_importance_scaled'] = results_df['avg_importance']/(results_df['avg_importance'].max())
    return results_df.sort_values(by='avg_importance', ascending=False)