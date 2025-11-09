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

def predictive_r2(y_pred, y_true_dmat):
    '''
    Function to compute the predicted r-squared given predictions and the true y.
    Input to the xgboost.cv() call in our hyperparameter search strategy, using this
    as a custom evaluation metric.

    Inputs:
        y_pred (np.array): the predictions of the y variable
        y_true_dmat (xgb.DMatrix): contains the true y values as labels in this xgb.DMatrix
    '''
    y_true = y_true_dmat.get_label()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Handle the case where ss_tot is zero (e.g., constant y_true)
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)

    return 'r2', r2

def ols_pred(x_cols, y_col, load_training_path, kf, center_response=False):
    '''
    Function to get the average predictive R-squared using OLS. Used to as a baseline with which
        to compare the performance of xgboost in the our hyperparameter selection procedure.

    Inputs:
        x_cols (list of strings): a list of the columns names to use as x-variables
        y_col (str): the column name of the desired outcome variable
        load_training_path (str): the path where the training data can be accessed from
        kf (sklearn.ModelSelection.KFold): a k-fold object specifying how the data will be split for CV
        center_response (bool): whether or not the response variable should be centered, defaults to false
    Outputs:
        avg_val_r2 (float): the average validation error across folds in CV
    '''

    train_data = pd.read_csv(load_training_path, index_col='Label')
    X = train_data[x_cols]
    y = train_data[y_col]

    if center_response:
        y = y - y.mean()
    
    pred_r2_folds = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        y_true_dmat = xgb.DMatrix(X_test, label=y_test)

        pred_r2_folds.append(predictive_r2(y_pred, y_true_dmat)[1])

    avg_val_r2 = np.mean(pred_r2_folds)

    return float(avg_val_r2)

def kfold_cv(params, ):
    

def process_hyperparam_chunk(lst, etas, booster, tree_method, nrounds, early_stopping_rounds, kf, x_cols, y_col, load_training_path, center_response=False):
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
        center_resposne (bool): whether the response variable should be mean-centered, defaults to False
    Ouputs:
        A pandas.DataFrame where each row is a different combo of hyperparams in this chunk, and includes
        the average predictive R-squared for that chunk from CV.
    '''

    num_eta = len(etas)
    results_lst = np.zeros((len(etas)*len(lst), len(lst[0]) + 3))

    train_data = pd.read_csv(load_training_path, index_col='Label')
    X = train_data[x_cols]
    y = train_data[y_col]

    if center_response:
        y = y - y.mean()

    columns = ['gamma', 'max_depth', 'reg_lambda', 'min_child_weight', 'eta', 'num_boost', 'test_rmse']

    early_stopping_callback = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name='r2',
            maximize=True)
    
    for i, param_set in enumerate(lst):
        for j, eta in enumerate(etas):
            params = dict(booster=booster,
                   eta=eta,
                   gamma=param_set[0],
                   max_depth=param_set[1],
                   reg_lambda=param_set[2],
                   min_child_weight=param_set[3],
                   tree_method=tree_method)

            dtrain = xgb.DMatrix(X, label=y)
            
            res = xgb.cv(params,
                   dtrain,
                   nrounds,
                   seed=12345,
                   folds=kf,
                   custom_metric=predictive_r2,
                   callbacks=[early_stopping_callback])
            results_lst[i*num_eta + j,:] = np.array(list(param_set) + [eta] + [res.shape[0]] + [float(res['test-r2-mean'].iloc[-1])])

    results_df = pd.DataFrame(results_lst, columns=columns)

    return results_df