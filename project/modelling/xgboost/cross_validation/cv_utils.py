import pandas as pd
import xgboost as xgb
import numpy as np

def process_hyperparam_chunk(lst, etas, booster, tree_method, nrounds, early_stopping_rounds, x_cols, y_col, load_training_path):
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
    Ouputs:
        A pandas.DataFrame with the hyperparam set and the averaged rmse on the test set.
    '''

    num_eta = len(etas)
    results_lst = np.zeros((len(etas)*len(lst), len(lst[0]) + 3))

    train_data = pd.read_csv(load_training_path, index_col='Label')
    X = train_data[x_cols]

    columns = ['gamma', 'max_depth', 'reg_lambda', 'min_child_weight', 'eta', 'num_boost', 'test_rmse']
    
    for i, param_set in enumerate(lst):
        for j, eta in enumerate(etas):
            params = dict(booster=booster,
                   eta=eta,
                   gamma=param_set[0],
                   max_depth=param_set[1],
                   reg_lambda=param_set[2],
                   min_child_weight=param_set[3],
                   tree_method=tree_method)

            dtrain = xgb.DMatrix(X, label=train_data[y_col])
            
            res = xgb.cv(params,
                   dtrain,
                   nrounds,
                   nfold=5,
                   seed=12345,
                   early_stopping_rounds=early_stopping_rounds)
            results_lst[i*num_eta + j,:] = np.array(list(param_set) + [eta] + [res.shape[0]] + [float(res['test-rmse-mean'].iloc[-1])])

    results_df = pd.DataFrame(results_lst, columns=columns)

    return results_df