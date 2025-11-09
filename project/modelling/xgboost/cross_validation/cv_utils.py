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
from sklearn.metrics import mean_squared_error
import ray

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

def ols_pred(x_cols, y_col, load_training_path, kf):
    '''
    Function to get the average predictive R-squared using OLS. Used to as a baseline with which
        to compare the performance of xgboost in the our hyperparameter selection procedure.

    Inputs:
        x_cols (list of strings): a list of the columns names to use as x-variables
        y_col (str): the column name of the desired outcome variable
        load_training_path (str): the path where the training data can be accessed from
        kf (sklearn.ModelSelection.KFold): a k-fold object specifying how the data will be split for CV
    Outputs:
        avg_val_r2 (float): the average validation error across folds in CV
    '''

    train_data = pd.read_csv(load_training_path, index_col='Label')
    X = train_data[x_cols]
    y = train_data[y_col]
    
    pred_r2_folds = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_train_centered = y_train - y_train.mean()
    
        reg = LinearRegression().fit(X_train, y_train_centered)
        y_pred = reg.predict(X_test) + y_train.mean()

        pred_r2_folds.append(predictive_r2(y_pred, y_test))

    avg_val_r2 = np.mean(pred_r2_folds)

    return float(avg_val_r2)

def kfold_cv(params, nrounds, early_stopping_rounds, kf, X, y):
    '''
    A function that mimics the behavior of xgb.cv(), but allows us to center the response
        variables with each training fold's mean in each iteration of k-fold CV.

    params (dict): 
    '''

    fold_train_means = []
    y_val_original = []
    dtrain_list = []
    dval_list = []
    model_list = []
    # first, build train and validation datasets, with training mean-subtracted responses
    # build a model for each train/validation split in 5 fold CV
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        y_val_original.append(y_val)
        
        y_train_mean_fold = np.mean(y_train)
        fold_train_means.append(y_train_mean_fold)
        
        y_train_centered = y_train - y_train_mean_fold
        y_val_centered = y_val - y_train_mean_fold
        
        dtrain_fold = xgb.DMatrix(X_train, label=y_train_centered)
        dval_fold = xgb.DMatrix(X_val, label=y_val_centered)
        
        dtrain_list.append(dtrain_fold)
        dval_list.append(dval_fold)

        model = xgb.Booster(
            params,
            [dtrain_fold, dval_fold]
        )
        model_list.append(model)

    # instantiating variables to track performance as trees are incrementally added
    best_avg_r2 = -np.inf
    best_iteration = 0
    consecutive_no_improve = 0
    
    # store results from the best iteration
    best_indiv_r2_scores = [0]*kf.get_n_splits()

    for i in range(nrounds):
        r2_scores_round = []
        rmse_scores_round = []
        # for each of k many train-validation split of the data
        for k in range(kf.get_n_splits()):
            model_list[k].update(dtrain_list[k], i)

            # get predictions for this model on the validation set
            y_pred_centered = model_list[k].predict(dval_list[k])

            # calculate the R-squared
            y_true_original = y_val_original[k]
            y_pred_original = y_pred_centered+fold_train_means[k]
            
            r2 = predictive_r2(y_pred_original, y_true_original)
            r2_scores_round.append(r2)
            rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
            rmse_scores_round.append(rmse)
        avg_r2_round = np.mean(r2_scores_round)

        # check early stopping conditions
        if avg_r2_round > best_avg_r2:
            best_avg_r2 = avg_r2_round
            best_iteration = i
            best_indiv_r2_scores = r2_scores_round
            best_indiv_rmse_scores = rmse_scores_round
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

        # if we've gone long enough with no improvement in average R-squared
        if consecutive_no_improve >= early_stopping_rounds:
            break
            
    return {'n_boost': best_iteration,
            'val-r2-mean': best_avg_r2,
            'val-r2-std': np.std(best_indiv_r2_scores),
            'val-rmse-mean': np.mean(best_indiv_rmse_scores),
            'val-rmse-std': np.std(best_indiv_rmse_scores)}

def process_hyperparam_chunk(lst, etas, booster, tree_method, nrounds, early_stopping_rounds, kf, x_cols, y_col, load_training_path):
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
        x_cols (list of str): the columns to use in the predictor space
        y_col (str): the column to use as the outcome
        load_training_path (str): path to the training data
        
    Ouputs:
        A pandas.DataFrame where each row is a different combo of hyperparams in this chunk, and includes
        the average predictive R-squared for that chunk from CV.
    '''

    num_eta = len(etas)
    results_lst = np.zeros((len(etas)*len(lst), len(lst[0]) + 4))

    train_data = pd.read_csv(load_training_path, index_col='Label')
    X = train_data[x_cols]
    y = train_data[y_col]

    columns = ['gamma', 'max_depth', 'reg_lambda', 'min_child_weight', 'eta', 'num_boost', 'val_rmse_mean', 'val_r2_mean']

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
                   tree_method=tree_method,
                   objective='reg:squarederror',
                   eval_metric='rmse')

            cv_res = kfold_cv(params, nrounds, early_stopping_rounds, kf, X, y)
            
            
            results_lst[i*num_eta + j,:] = np.array(list(param_set) + [eta, cv_res['n_boost'], cv_res['val-rmse-mean'], cv_res['val-r2-mean']])

    results_df = pd.DataFrame(results_lst, columns=columns)

    return results_df
