'''
Utility functions to be used in the hyperparameter search procedure for the xgboost models.

Jimmy Butler
October 2025
'''

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def get_shrinkage_factor(y_pred, y_true):
    '''
    Helper function to implement predictive shrinkage. Returns
        a shrinkage factor.

    Inputs:
        y_pred (np.array): the predictions we wish to shrink
        y_true (np.array): the true y's (usually heldout y's in training)

    Outputs:
        shrinkage_factor (float): the shrinkage factor
    '''

    ols = LinearRegression(fit_intercept=False)
    fit = ols.fit(X=y_pred.reshape(-1,1), y=y_true)
    shrinkage_factor = fit.coef_[0]

    return shrinkage_factor

def predictive_r2(y_pred, y_true):
    '''
    Function to compute the predicted r-squared given predictions and the true y.
    Input to the xgboost.cv() call in our hyperparameter search strategy, using this
    as a custom evaluation metric.

    Inputs:
        y_pred (np.array): the predictions of the y variable
        y_true (np.array): the true y's we wish to capture
    '''
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Handle the case where ss_tot is zero (e.g., constant y_true)
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)

    return r2

def ols_pred(data_CV_splits, shrink=False):
    '''
    Function to get the average predictive R-squared using OLS. Used to as a baseline with which
        to compare the performance of xgboost in the our hyperparameter selection procedure.

    Inputs:
        data_CV_splits (list of dicts): for each train/validation split of data, stores validation predictors and
            outcomes to be used for training and evaluation, as well as predictions from intermediate models (like CLA for snow)
        shrink (bool): whether to use predictive shrinkage
    Outputs:
        avg_val_r2 (float): the average validation error across folds in CV
    '''

    pred_folds = []
    val_folds = []
    for split in data_CV_splits:
        X_train, X_val = split['X_train'], split['X_val']
        y_train_centered, y_val_centered = split['y_train_centered'], split['y_val_centered']
    
        reg = LinearRegression().fit(X_train, y_train_centered)
        y_pred = reg.predict(X_val)
        pred_folds.append(y_pred)
        val_folds.append(y_val_centered)

    shrinkage_factor = 1

    if shrink:
        pred_folds_flat = np.concatenate(pred_folds)
        val_folds_flat = np.concatenate(val_folds)
        shrinkage_factor = get_shrinkage_factor(pred_folds_flat, val_folds_flat)

    pred_r2_folds = []

    for k in range(len(data_CV_splits)):
        pred_r2_folds.append(predictive_r2(shrinkage_factor*pred_folds[k], val_folds[k]))

    avg_val_r2 = np.mean(pred_r2_folds)

    return float(avg_val_r2)

def kfold_cv(params, nrounds, early_stopping_rounds, data_CV_splits, shrink):
    '''
    A function that mimics the behavior of xgb.cv(), but allows us to center the response
        variables with each training fold's mean in each iteration of k-fold CV.

    params (dict): 
    '''

    # flag to see if residuals are needed
    using_lm = ('lm_val_preds' in data_CV_splits[0].keys())

    # by default, xgb will be trained on centered outcome, unless
    # we must train on residuals instead
    y_lab = 'y_train_centered'
    if using_lm:
        y_lab = 'y_train_resid'
    
    dtrain_lst = []
    dval_lst = []
    xgbmodel_lst = []

    for split in data_CV_splits:
        
        dtrain_fold = xgb.DMatrix(split['X_train'], label=split[y_lab])
        dval_fold = xgb.DMatrix(split['X_val'], label=split['y_val_centered'])
        
        dtrain_lst.append(dtrain_fold)
        dval_lst.append(dval_fold)

        model = xgb.Booster(
            params,
            [dtrain_fold, dval_fold]
        )
        xgbmodel_lst.append(model)

    # instantiating variables to track performance as trees are incrementally added
    best_avg_r2 = -np.inf
    best_iteration = 0
    consecutive_no_improve = 0
    
    # store results from the best iteration
    best_indiv_r2_scores = [0]*len(data_CV_splits)

    for i in range(nrounds):
        r2_scores_round = []
        rmse_scores_round = []
        # for each of k many train-validation split of the data
        y_pred_folds = []
        y_val_folds = []
        for k in range(len(data_CV_splits)):
            
            data_CV_split = data_CV_splits[k]
            xgbmodel_lst[k].update(dtrain_lst[k], i)

            y_pred_centered = xgbmodel_lst[k].predict(dval_lst[k])
            # if there are linear model preds to consider, add to the predictions
            if using_lm:
                y_pred_centered = data_CV_split['lm_val_preds'] + y_pred_centered

            y_pred_folds.append(y_pred_centered)
            y_val_folds.append(data_CV_split['y_val_centered'])
            

        shrinkage_factor = 1 # by default, no predictive shrinkage
        if shrink:
            y_pred_flat = np.concatenate(y_pred_folds)
            y_val_flat = np.concatenate(y_val_folds)
            shrinkage_factor = get_shrinkage_factor(y_pred_flat, y_val_flat)

        for k in range(len(data_CV_splits)):
            
            data_CV_split = data_CV_splits[k]

            y_val = y_val_folds[k]
            y_pred = y_pred_folds[k]*shrinkage_factor
            # calculate the R-squared     
            r2 = predictive_r2(y_pred, y_val)
            r2_scores_round.append(r2)
            rmse = np.sqrt(np.mean((y_pred - y_val)**2))
            rmse_scores_round.append(rmse)
        avg_r2_round = np.mean(r2_scores_round)

        # check early stopping conditions
        if avg_r2_round > best_avg_r2:
            best_avg_r2 = avg_r2_round
            best_iteration = i
            best_shrinkage = shrinkage_factor
            best_indiv_r2_scores = r2_scores_round
            best_indiv_rmse_scores = rmse_scores_round
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

        # if we've gone long enough with no improvement in average R-squared
        if consecutive_no_improve >= early_stopping_rounds:
            break
            
    return {'n_boost': best_iteration,
            'shrinkage_factor': best_shrinkage,
            'val-r2-mean': best_avg_r2,
            'val-r2-std': np.std(best_indiv_r2_scores),
            'val-rmse-mean': np.mean(best_indiv_rmse_scores),
            'val-rmse-std': np.std(best_indiv_rmse_scores)}

def process_hyperparam_chunk(lst, etas, booster, tree_method, nrounds, early_stopping_rounds, data_CV_splits, shrink, seed):
    '''
    A helper function for xgboost hyperparameter selection procedure. Given a list of combinations of
        all other hyperparams other than the learning rate and number of boosting rounds, as well
        as a list of learning rates, conduct a 5-fold CV on each hyperparam combo, with early stopping.
        Intended to be executed in parallel with other lists of hyperparams in the cross_validation_xgb.py
        script.

    Inputs:
        lst (list): the list of lists of all other hyperparams (in order: gamma, max tree depth, lambda, min child weight)
        etas (list): the list of learning rates to try out given the other hyperparam combo
        booster (string): the booster method for the xgb software
        tree_method (string): method to use to construct trees (usually 'exact')
        nrounds (int): number of boosting rounds
        early_stopping_rounds (int): number of rounds to check for improvement in validation error
        kf (sklearn.ModelSelection.KFold): a k-fold object specifying how the data will be split for CV
        data_CV_splits (list of dicts): for each train/validation split of data, stores validation predictors and
            outcomes to be used for training and evaluation, as well as predictions from intermediate models (like CLA for snow)
        
    Ouputs:
        A pandas.DataFrame where each row is a different combo of hyperparams in this chunk, and includes
        the average predictive R-squared for that chunk from CV.
    '''

    num_eta = len(etas)
    results_lst = np.zeros((len(etas)*len(lst), len(lst[0]) + 5))

    columns = ['gamma', 'max_depth', 
               'reg_lambda', 'min_child_weight', 
               'subsample_frac', 'eta', 
               'num_boost', 'shrinkage_factor', 
               'val_rmse_mean', 'val_r2_mean']

    # for every set of "other" hyperparams
    for i, param_set in enumerate(lst):
        # for every learning rate, fixing other set of hyperparams
        for j, eta in enumerate(etas):
            params = dict(booster=booster,
                   eta=eta,
                   gamma=param_set[0],
                   max_depth=param_set[1],
                   reg_lambda=param_set[2],
                   min_child_weight=param_set[3],
                   subsample=param_set[4],
                   tree_method=tree_method,
                   objective='reg:squarederror',
                   eval_metric='rmse',
                   seed=seed)

            cv_res = kfold_cv(params, nrounds, early_stopping_rounds, data_CV_splits, shrink)
            
            
            results_lst[i*num_eta + j,:] = np.array(list(param_set) + [eta, 
                                                                       cv_res['n_boost'], 
                                                                       cv_res['shrinkage_factor'], 
                                                                       cv_res['val-rmse-mean'], 
                                                                       cv_res['val-r2-mean']])

    results_df = pd.DataFrame(results_lst, columns=columns)

    return results_df
