'''
main predictive algorithm
'''

import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time
import pickle
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from skopt import BayesSearchCV
from catboost import Pool, cv

import utils_function
        
import preprocessing4
import importlib
import ipynb.fs.full.preprocessing3_smote
importlib.reload(ipynb.fs.full.preprocessing3_smote)

def catDefault(X_train, y_train):
    labelcount = y_train.value_counts()    
    cat_features = list(X_train.select_dtypes('bool').columns)
    cvmodel = CatBoostClassifier(scale_pos_weight=labelcount[0]/labelcount[1], 
                            objective='Logloss', eval_metric='AUC', verbose=50,
                            early_stopping_rounds=50, cat_features=cat_features,                                 
                            custom_metric=['Logloss', 'AUC:hints=skip_train~false'])
    return cvmodel

def catRandomSearch(X_train, y_train):
    labelcount = y_train.value_counts()    
    cat_features = list(X_train.select_dtypes('bool').columns)    
    cvmodel = CatBoostClassifier(scale_pos_weight=labelcount[0]/labelcount[1], 
                            objective='Logloss', eval_metric='AUC:hints=skip_train~false', verbose=50, 
                            early_stopping_rounds=50, cat_features=cat_features)
    params = {
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bylevel': [0.1, 0.5, 1.0],
            'max_depth': [5, 7, 16],
            'learning_rate': [0.1, 0.5],
            'n_estimators': [50, 200, 1000]
            }
    randomized_search_result = cvmodel.randomized_search(params, X=X_train, y=y_train, cv=5, n_iter=20)
    bestmodel = CatBoostClassifier(scale_pos_weight=labelcount[0]/labelcount[1], 
                               objective='Logloss', eval_metric='AUC:hints=skip_train~false', verbose=50, 
                                early_stopping_rounds=50, cat_features=cat_features, **randomized_search_result['params'])
    return bestmodel

def runxgboost(configs_variables, returnflag=False, X_train=None, X_test=None, y_train=None, y_test=None):
    
    year=3000
#    configs_variables = utils_function.read_config(site)
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    stg = configs_variables['stg']
    fs = configs_variables['fs']
    oversample = configs_variables['oversample']
    model_type = configs_variables['model_type']
    drop_correlation_catboost = configs_variables['drop_correlation_catboost']
        
    print('Running '+model_type+' on site '+site+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
    
    if drop_correlation_catboost:
        suffix='nc'
    else:
        suffix= ''
    
    #load tables
    if X_train is None:
        X_train = pd.read_pickle(datafolder+site+'/X_train_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        X_test =  pd.read_pickle(datafolder+site+ '/X_test_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        y_train = pd.read_pickle(datafolder+site+'/y_train_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        y_test =  pd.read_pickle(datafolder+site+ '/y_test_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')

    X_train = X_train.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
    X_test = X_test.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
        
        
    tic = time.perf_counter()     

    #catboost
    if model_type == "catd":
        bestmodel = catDefault(X_train, y_train)
    if model_type == "catr":
        bestmodel = catRandomSearch(X_train, y_train)

    print('Training xgb/cat on site '+site+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
    bestmodel.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=50, early_stopping_rounds=50)
    prelabel = bestmodel.predict(X_test)

    pred = bestmodel.predict_proba(X_test)
    roc = roc_auc_score(y_test, pred[:,1])    
    
    print('roc = '+ str(roc))
    print('Confusion Matrix')
    cm = confusion_matrix(y_test, prelabel)
    print(cm)
    
    toc = time.perf_counter()
    print('Finished '+model_type+' on site '+site+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)        
    print(f"{site}:{year}:{stg}:{fs}:{oversample}: finished in {toc - tic:0.4f} seconds")  
    if returnflag:
        return bestmodel, roc, cm
    pickle.dump(bestmodel, open(datafolder+site+'/model_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl', 'wb'))    
    
#    pickle.dump(bestmodel, open(datafolder+site+'/model_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl', 'wb'))        
    
def boosttrapcatboost(configs_variables, numberbt):      

    '''
    This module run on cross validation dataset
    '''    
    year = 3000
#    configs_variables = utils_function.read_config(site)
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    fs = configs_variables['fs']
    stg = configs_variables['stg']
    oversample = configs_variables['oversample']
    model_type = configs_variables['model_type']
    
    drop_correlation_catboost = configs_variables['drop_correlation_catboost']    
    n_splits = int(configs_variables['n_splits'])
    random_state = int(configs_variables['random_state'])
    
    if drop_correlation_catboost:
        suffix = 'nc'
    else:
        suffix = ''      
    
    print('Training BT ' +str(numberbt)+ ' cat on site '+site+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
    
    X_train, X_test, y_train, y_test = ipynb.fs.full.preprocessing3_smote.get_boosttrap(configs_variables, numberbt)
    
    bestmodel, roc, cm = runxgboost(configs_variables, returnflag=True, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)       
    
    saveobjpkl = (configs_variables, numberbt, bestmodel, roc, cm)
    pickle.dump(saveobjpkl, open(datafolder+site+'/boosttrap_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'_'+str(numberbt)+'.pkl', 'wb'))    
    
    print(datafolder+site+'/boosttrap_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'_'+str(numberbt)+'.pkl')
    