import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
import time
from sklearn.model_selection import RandomizedSearchCV
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
from collections.abc import Iterable
import math
import random
from scipy.stats import spearmanr, pearsonr
import joblib
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import shap
import statsmodels.api as sm


shap.initjs()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



def result_preprocess(datafilepath,desired_cols, n_rows=None):
    results = pd.read_csv(datafilepath)

    if n_rows is None:
        data_m = results[desired_cols]
        # label_m = results[['protein_rate']]
        label_m = results[['titer(mg/L)']]
    else:
        data_m = results[desired_cols].iloc[:n_rows, :]        
        label_m = results[['titer(mg/L)']].iloc[:n_rows, :]
    return data_m, label_m


def training_model(datafilepath,outputpath):    
    desired_cols = ['(NH4)2SO4','Triton X-100','Glycine',
                    'CSL-P','NaCl','K2HPO4','Tryptone',
                    'YE','Methionine','Cysteine','NH4OAc',
                    'Glycerol','Na2S2O3','ZnSO4·7H2O','MgSO4·7H2O','FAC']   
    aggregated_data_m = pd.DataFrame(columns=desired_cols)    
    aggregated_label_m = pd.DataFrame(columns=['titer(mg/L)'])
    # aggregated_label_m = pd.DataFrame(columns=['protein_rate'])
    data_m,label_m= result_preprocess(datafilepath, desired_cols, len(pd.read_csv(datafilepath)))
    aggregated_data_m = pd.concat([aggregated_data_m, data_m]).reset_index(drop=True)
    aggregated_label_m = pd.concat([aggregated_label_m, label_m]).reset_index(drop=True)

    model = XGBRegressor(objective = 'reg:squarederror')
    # Create the grid search parameter and scoring functions
    param_grid = {
        "learning_rate": [0.01, 0.03, 0.1, 0.3],
        "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
        "subsample": [0.6, 0.8, 0.9, 1.0],
        "max_depth": [2, 3, 4, 6 , 8],
        "n_estimators": [10, 20,  40, 60, 80, 100, 300, 500],
        "reg_lambda": [1, 1.5, 2],
        "gamma": [0, 0.1, 0.4, 0.6],
        "min_child_weight": [1, 2, 4]}   
    grid = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid,
        cv=5,
        scoring= 'neg_mean_absolute_error',
        n_jobs=-1,
        n_iter=200) 

    print('RandomSearchCV ...')

    grid.fit(aggregated_data_m.values, aggregated_label_m.values)
    results = pd.DataFrame(grid.cv_results_).sort_values('mean_test_score', ascending=False)    
    ensemble_len = 20
    regressors_list = [
        XGBRegressor(objective='reg:squarederror', **param)
            .fit(aggregated_data_m.values, aggregated_label_m.values.ravel())
        for param in results.params.iloc[:ensemble_len]
    ]
    joblib.dump(regressors_list, outputpath)
    print('RandomSearchCV Done!')


def bayesian_optimization(regressors_list_name,  
                          randomDataSetPath, 
                          lastrounddataPath,                                                
                          exploitation=1, exploration=1, test_size=100,                                                
                          ):    
   
    regressors_list = joblib.load(regressors_list_name)    
    df_1 = pd.read_csv(randomDataSetPath)
    desired_cols = list(df_1.columns)
    final_order = desired_cols

    aggregated_data_m = pd.DataFrame(columns=desired_cols)
    data_m, label_m= result_preprocess(lastrounddataPath, desired_cols, len(pd.read_csv(lastrounddataPath)))
    aggregated_data_m = pd.concat([aggregated_data_m, data_m]).reset_index(drop=True)
    df_main = aggregated_data_m

    df_temp = df_1.copy(deep=True)
    for index, regressor in enumerate(regressors_list):
        df_1['pred_titer_{}'.format(index)] = regressor.predict(df_temp.values)

    df_1['regressors_std'] = df_1[[str(i) for i in df_1.columns if 'pred_titer' in str(i)]].std(axis=1)
    df_1['mean_vote'] = df_1[[str(i) for i in df_1.columns if 'pred_titer' in str(i)]].mean(axis=1)
    df_1['UCB'] = exploitation * df_1['mean_vote'] + exploration * df_1['regressors_std']
    df_1 = df_1.sort_values(['UCB'], ascending=False)
       
    return df_1


PROJECT_ROOT = Path.cwd()
# need to change the path to your own data path
ROUND = 'Round1'
TRAIN_DATA = PROJECT_ROOT/'Example'/ROUND/"train_data_R1-0+R1-1.csv"

SAVE_DIR = PROJECT_ROOT/'Example'/ROUND/'output'
SAVE_DIR.mkdir(exist_ok=True)

training_model(TRAIN_DATA,SAVE_DIR/f"model_{ROUND}.joblib")

df_result = bayesian_optimization(SAVE_DIR/f"model_{ROUND}.joblib",SAVE_DIR/"value_combination.csv",TRAIN_DATA)  
df_result.to_csv(SAVE_DIR/f"R2_virtual_recipe_{ROUND}.csv", index=False)  

print("successfully finished")
