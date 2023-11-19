from tkinter import Grid
import numpy as np
import pandas as pd
from typing import Dict, List
import math
import scipy
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, ExpSineSquared, Matern, ConstantKernel, WhiteKernel # , Polynomial
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer


def get_mean_based_on_season(df: pd.DataFrame): 
    df_mean = df.groupby(["season"]).mean()
    return df_mean.to_dict(orient="index")

def replace_nans_with_mean(df: pd.DataFrame):
    season_to_dict = get_mean_based_on_season(df)
    for i in df.index:
        season = df.at[i, "season"]
        for col in df.columns:
            if col == "season":
              continue
            if math.isnan(df.at[i, col]):
                season_dict = season_to_dict[season]
                mean = season_dict[col]
                df.at[i, col] = mean
    return df

def encode_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical data using principles of the OrdinalEncoder()"""
    cat_dict_season = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
    df = df.replace(
        {
            "season": cat_dict_season,
        }
    )
    return df



def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
   
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    train_df = replace_nans_with_mean(train_df)
    test_df = replace_nans_with_mean(test_df)


    train_df = encode_categorical_data(train_df)
    test_df = encode_categorical_data(test_df)

    y_train = train_df['price_CHF']
    X_train = train_df.drop(columns=['price_CHF'])
    X_test = test_df

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"

    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    parameters = {'kernel':(DotProduct(), RBF(), Matern(), RationalQuadratic(), Matern(), ConstantKernel())} # TODO: Try ExpSineSquared later because alpha crashed
    gpr = GaussianProcessRegressor()
    scorer = make_scorer(r2_score)
    gs_results = GridSearchCV(gpr, parameters, scoring=scorer, n_jobs=3, refit=True, verbose=3, error_score='raise', cv=5)
    gs_results.fit(X_train, y_train)
    print(gs_results.best_params_)
    print(gs_results.best_score_)
    y_pred = gs_results.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

if __name__ == "__main__":
    X_train, y_train, X_test = data_loading()
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

