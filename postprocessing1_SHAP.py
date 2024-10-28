import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import BSpline, make_interp_spline, interp1d
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr
import csv
from dfply import *
from xgboost import XGBClassifier
import itertools
import os
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import pickle
from glob import glob
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from catboost import Pool, cv
import xgboost
import catboost
import scipy.stats as st
import utils_function

def collectSHAP_sub(site, year, stg, fs, oversample, model_type, returnflag=False):
# site = 'MCRI'
# year = 3000
# stg = 'stg23'
# fs = 'nofs'
# oversample = 'raw'
# model_type = 'catd'
# ckd_group=0
# returnflag=False
#if True:
    print('Running shap '+model_type+' on site '+site+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
    tic = time.perf_counter()     

    #load model
    print('Running cross_roc '+model_type+' on site '+site_m+'/'+site_d+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
    model = pickle.load(open(datafolder+site_m+'/model_'+model_type+'_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'.pkl', 'rb'))

    #load tables
    X_train = pd.read_pickle(datafolder+site+'/X_train_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    X_test =  pd.read_pickle(datafolder+site+'/X_test_' +site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    y_train = pd.read_pickle(datafolder+site+'/y_train_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    y_test =  pd.read_pickle(datafolder+site+'/y_test_' +site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')

    X_train = X_train.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
    X_test = X_test.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)    
    
    # Get AUC
#    pred = model.get_booster().predict(dtest, pred_contribs=False)
#    pred = model.predict(X_test)    
#    roc = roc_auc_score(y_test, pred)    

#     if ckd_group != 0:
# #        pass
#         return
#        X_train = X_train[X_train_ckdg['CKD_group']==ckd_group]
#        X_test = X_test[X_test_ckdg['CKD_group']==ckd_group]
#        y_train = y_train[X_train_ckdg['CKD_group']==ckd_group]
#        y_test = y_test[X_test_ckdg['CKD_group']==ckd_group]

    pred = model.predict_proba(X_test)
    roc = roc_auc_score(y_test, pred[:,1])       
    
    shapX = pd.concat([X_train, X_test])
    shapy = pd.concat([y_train, y_test])
    
    # Calculate SHAP value
    if type(model) == xgboost.sklearn.XGBClassifier:
        dshap  = xgb.DMatrix(shapX, label=shapy)
        shap = model.get_booster().predict(dshap, pred_contribs=True)
        # Get feature importance
        model_data = pd.concat([pd.DataFrame(model.get_booster().get_score(importance_type='cover'), index=['Cover']), \
        pd.DataFrame(model.get_booster().get_score(importance_type='gain'), index=['Gain']), \
        pd.DataFrame(model.get_booster().get_score(importance_type='weight'), index=['Frequency'])]).transpose() >> mutate(Feature = X.index)
        model_data['rank'] = model_data['Gain'].rank(method='min', ascending=False)
        used_feature = list(model.get_booster().get_score().keys())        
    elif type(model) == catboost.core.CatBoostClassifier:
        cat_features = model.get_cat_feature_indices()
        pshap = Pool(data=shapX, label=shapy, cat_features=cat_features)        
        shap = model.get_feature_importance(data=pshap, type='ShapValues')
        model_data = model.get_feature_importance(prettified=True)
        model_data['Feature'] = model_data['Feature Id']
        model_data = model_data >> select('Feature', 'Importances')
        model_data['rank'] = model_data['Importances'].rank(method='min', ascending=False)     
        used_feature = list((model_data >> mask(X.Importances!=0)).Feature)
    else:
    #Using shap package example
        import shap
        explainer = shap.Explainer(model, algorithm='permutation')
        shap_valuesX = explainer.shap_values(shapX)
        #shap.summary_plot(shap_valuesX, X_test, plot_type="bar")    
        shap = shap_valuesX    

    
    # Collect SHAP value
    def CI95(data):
        if len(data) == 1:
            return (np.nan, np.nan)
        return (np.nan, np.nan)            
#        return st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) #95% confidence interval

    shap_data = list()
    shap_data_raw = list()
    for i in range(shapX.columns.shape[0]):
        df = pd.DataFrame(list(zip(shapX.iloc[:,i], shap[:, i], abs(shap[:, i]))),columns =['Name', 'val', 'absval'])
        # Check confidence interval for one data point
        plot_data = df.groupby("Name").agg([np.mean, np.var, np.std, np.median, CI95, 'size']).reset_index()
        df.index = shapX.index

        plot_data_all = df.groupby("Name").agg([np.mean, np.var, np.std, np.median, CI95, 'size']).reset_index()
        plot_data_0= df[shapy==0].groupby("Name").agg([np.mean, np.var, np.std, np.median, CI95, 'size']).reset_index()
        plot_data_1= df[shapy==1].groupby("Name").agg([np.mean, np.var, np.std, np.median, CI95, 'size']).reset_index()

        plot_data_all.columns = [''.join(x) for x in plot_data_all.columns]
        plot_data_0.columns = [x+'_0' for x in plot_data_all.columns]
        plot_data_1.columns = [x+'_1' for x in plot_data_all.columns]
        plot_data_all = plot_data_all.drop('absvalsize', axis=1)
        plot_data_0   = plot_data_0.drop('absvalsize_0', axis=1)
        plot_data_1   = plot_data_1.drop('absvalsize_1', axis=1)
        plot_data_0 = plot_data_0.rename({'Name_0':'Name'},axis=1)
        plot_data_1 = plot_data_1.rename({'Name_1':'Name'},axis=1)
        plot_data = pd.merge(plot_data_all, plot_data_0, left_on='Name', right_on='Name', how='left')
        plot_data = pd.merge(plot_data, plot_data_1, left_on='Name', right_on='Name', how='left')        
        
        plot_data = plot_data >> mutate(Feature=shapX.columns[i])
        plot_data.columns = [''.join(x) for x in plot_data.columns]
        plot_data[['valCI95down', 'valCI95up']] = pd.DataFrame(plot_data['valCI95'].tolist(), index=plot_data.index)
        plot_data[['absvalCI95down', 'absvalCI95up']] = pd.DataFrame(plot_data['absvalCI95'].tolist(), index=plot_data.index)
        plot_data = plot_data.drop(['valCI95', 'absvalCI95'],axis=1)
        shap_data.append(plot_data.copy())        
        plot_data_raw = df >> select(X.Name, X.val) >> mutate(Feature=shapX.columns[i])        
        shap_data_raw.append(plot_data_raw.copy())
    shap_data = pd.concat(shap_data)
    shap_data_raw = pd.concat(shap_data_raw)    
#    shap_data= shap_data[shap_data['Feature'].isin(used_feature)]

    # create csv for metaregression
    shap_data = shap_data >> left_join(model_data, by='Feature')
    siteyr = site+'_'+model_type+'_'+fs+'_'+stg+'_'+oversample+'_'+'005'+"_"+str(year)    
    shap_data = shap_data >> mutate(siteyr=siteyr) >> rename(fval=X.Name) >> rename(mean_val=X.valmean) >> rename(se_val=X.valstd) >> rename(mean_imp = X.absvalmean) >> rename(se_imp = X.absvalstd) >> rename(var_imp = X.absvalvar) >> rename(median_val = X.valmedian) >> rename(median_imp = X.absvalmedian) >> rename(var_val = X.valvar)
    shap_data['site'] = site
    shap_data['year'] = year
    shap_data['stg'] = stg
    shap_data['fs'] = fs
    shap_data['oversample'] = oversample
    shap_data['model'] = model_type
    shap_data['rmcol'] = '005'
    
    # Calculate ranking base on absolute mean value of SHAP
    rank_abs_shap_max = (shap_data >> mutate(abs_shap_max = abs(X.mean_val))).loc[:,['Feature', 'abs_shap_max']].groupby(['Feature']).agg(np.max).reset_index()
    rank_abs_shap_max['rank_abs_shap_max'] = rank_abs_shap_max['abs_shap_max'].rank(method='min', ascending=False)
    shap_data = pd.merge(shap_data, rank_abs_shap_max, left_on=['Feature'], right_on=['Feature'], how='left')

    #Calculate ranking base on SHAP min max difference and variance
    tdata = shap_data.loc[:,['Feature', 'mean_val']].groupby(['Feature']).agg([np.max,np.min,np.var]).reset_index()
    tdata.columns = ['Feature', 'maxSHAP', 'minSHAP', 'varSHAP']
    tdata = (tdata >> mutate(minmax_SHAP = X.maxSHAP-X.minSHAP))
    tdata['rank_minmax_SHAP'] = tdata['minmax_SHAP'].rank(method='min', ascending=False)
    tdata['rank_var_SHAP'] = tdata['varSHAP'].rank(method='min', ascending=False)
    shap_data = pd.merge(shap_data, tdata, left_on=['Feature'], right_on=['Feature'], how='left')    

    # add auc value
    shap_data = shap_data >> mutate(auc=roc)
    
    #sort
    shap_data = shap_data.sort_values(['rank', 'fval'])

    #calculate confusion matrix
    cdata = pd.concat([pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)], axis=0)
    cmdata = cdata.melt(id_vars='FLAG', value_vars= list(cdata.columns).remove('FLAG'))
    conmat = cmdata.groupby(['FLAG', 'variable','value']).size().reset_index()
    conmat2 = conmat.pivot(index=['variable', 'value'], columns='FLAG', values=0).fillna(0).reset_index()
    conmat2.columns = ['Feature', 'fval', 'b', 'a']    
    conmat3 = cmdata.groupby(['FLAG', 'variable']).size().reset_index()
    conmat4 = conmat3.pivot(index=['variable'], columns='FLAG', values=0).fillna(0).reset_index()
    conmat4.columns = ['Feature', 'd', 'c'] 
    conmat5 = pd.merge(conmat2, conmat4, left_on='Feature', right_on='Feature', how='left')
    conmat6 = conmat5 >> mutate(d=X.d-X.b) >> mutate(c=X.c-X.a) >> mutate(num=X.a+X.b)
    conmat6['fval'] = conmat6['fval'].astype('float64')
    shap_data = pd.merge(shap_data, conmat6, left_on=['Feature', 'fval'], right_on=['Feature', 'fval'], how='left')
    
    #is categorical?
    X_test =  pd.read_pickle(datafolder+site+ '/X_test_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    cat_features = pd.DataFrame(list(X_test.select_dtypes('bool').columns)) >> mutate(isCategorical = True)
    cat_features.columns = ['Feature', 'isCategorical']
    shap_data = pd.merge(shap_data, cat_features, right_on='Feature', left_on='Feature', how='left')
    shap_data.loc[:,'isCategorical'] = shap_data.loc[:,'isCategorical'].fillna(False)    
    
    #Collect fval range and stats
    Xdata = pd.concat([X_train, X_test], axis=0)
    try:
        filtertable = Xdata.select_dtypes(exclude=bool).agg([np.min, np.max, np.mean,np.std],axis=0).transpose()
        filtertable = filtertable.assign(upr=filtertable['mean']+3*filtertable['std']).assign(lwr=filtertable['mean']-3*filtertable['std']).reset_index().rename({'index':'Feature', 'mean':'fval_mean', 'std':'fval_std', 'upr':'fval_upr', 'lwr':'fval_lwr', 'amax':'fval_max', 'amin':'fval_min'},axis=1)
        shap_data  = pd.merge(shap_data, filtertable, right_on='Feature', left_on='Feature', how='left')
    except:
        pass
    
    #Save shap_data 
    if returnflag:
#        pass
        return shap_data, shap_data_raw
    else:
        shap_data.to_pickle(datafolder+site+'/shapdata_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'_'+str(ckd_group)+'_005.pkl')
        shap_data_raw.to_pickle(datafolder+site+'/shapdataraw_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'_'+str(ckd_group)+'_005.pkl')
    #model.to_pickle(datafolder+site+'/model_data_'+site+'_'+str(year)+'.pkl')

    toc = time.perf_counter()
    print(f"{site}:{year} finished in {toc - tic:0.4f} seconds")  
    print('Finished shap '+model_type+' on site '+site+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)    

#print('done')

def collectSHAP(site, year, stg, fs, oversample, model_type):
    shap_data_list = list()
    shap_data_raw_list = list()    
    for i in range(1,5):
        try:
            shap_data, shap_data_raw = collectSHAP_sub(site, year, stg, fs, oversample, model_type, ckd_group=i, returnflag=True)
        except Exception as error:
            print(site+":"+str(year)+":"+stg+":"+fs+":"+oversample+":"+model_type+" raised " + "error" +"\n"+error.traceback)        
        shap_data['ckd_group'] = i
        shap_data_raw['ckd_group'] = i
        shap_data_list.append(shap_data.copy())
        shap_data_raw_list.append(shap_data_raw.copy())
    shap_data_all = pd.concat(shap_data_list)
    shap_data_raw_all = pd.concat(shap_data_raw_list)
    shap_data_all.to_pickle(datafolder+site+'/shapdata_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'_005.pkl')
    shap_data_raw_all.to_pickle(datafolder+site+'/shapdataraw_'+model_type+'_'+site+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'_005.pkl')
    
def cross_roc(site_m, site_d, year, stg, fs, oversample, model_type, ckd_group=0, returnflag=False):

    model = pickle.load(open(datafolder+site_m+'/model_'+model_type+'_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+'.pkl', 'rb'))

    #load tables
    X_train_m = pd.read_pickle(datafolder+site_m+'/X_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    y_train_m = pd.read_pickle(datafolder+site_m+'/y_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    X_test_m = pd.read_pickle(datafolder+site_m+'/X_test_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    y_test_m = pd.read_pickle(datafolder+site_m+'/y_test_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')

    X_train_d = pd.read_pickle(datafolder+site_d+'/X_train_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    y_train_d = pd.read_pickle(datafolder+site_d+'/y_train_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    X_test_d =  pd.read_pickle(datafolder+site_d+'/X_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    y_test_d =  pd.read_pickle(datafolder+site_d+'/y_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')

    common_features = [x for x in X_test_d.columns if x in X_train_m.columns]

    X_train2_d = X_train_d[common_features]
    X_test2_d = X_test_d[common_features]

    X_train2_m = X_train_m.iloc[0:1]
    X_test2_m = X_test_m.iloc[0:1]

    X_train3_d = pd.concat([X_train2_m, X_train2_d]).iloc[1:]
    X_test3_d = pd.concat([X_test2_m, X_test2_d]).iloc[1:]

    X_train3_d.loc[:,X_train2_m.dtypes==bool] = X_train3_d.loc[:,X_train2_m.dtypes==bool].fillna(False)
    X_test3_d.loc[:,X_test2_m.dtypes==bool] = X_test3_d.loc[:,X_test2_m.dtypes==bool].fillna(False)

    pred = model.predict_proba(X_test3_d)

    roc = roc_auc_score(y_test_d, pred[:,1])
    return roc

def collectSHAP_cross_sub(configs_variable_m, configs_variable_d, returnflag=False):
    '''
    This function apply internal and external data to each cross validate model of site_m
    if site_d = site_m, the validaton set is used
    else all data from site_d is used
    '''

    year=3000
    site_m, datafolder, home_directory = utils_function.get_commons(configs_variable_m)
    site_d, datafolder, home_directory = utils_function.get_commons(configs_variable_d)
    
    datafolder = configs_variable_m['datafolder']
    stg = configs_variable_m['stg']
    fs = configs_variable_m['fs']
    oversample = configs_variable_m['oversample']
    model_type = configs_variable_m['model_type']   

    drop_correlation_catboost = configs_variable_m['drop_correlation_catboost']
    if drop_correlation_catboost:
        suffix = 'nc'
    else:
        suffix = ''     
    
    if not configs_variable_m['rerun_flag'] and os.path.exists(datafolder+site_m+'/shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet'):
        print('Existed: shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet')
        return        
    
    
    print('Running collectSHAP_cross_sub '+model_type+' on site '+site_m+'/'+site_d+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
    tic = time.perf_counter()     

    #load model
    model = pickle.load(open(datafolder+site_m+'/model_'+model_type+'_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl', 'rb'))

    #load tables
    X_train_m = pd.read_pickle(datafolder+site_m+'/X_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    y_train_m = pd.read_pickle(datafolder+site_m+'/y_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    X_test_m = pd.read_pickle(datafolder+site_m+'/X_test_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    y_test_m = pd.read_pickle(datafolder+site_m+'/y_test_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')

    X_train_d = pd.read_pickle(datafolder+site_d+'/X_train_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    y_train_d = pd.read_pickle(datafolder+site_d+'/y_train_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    X_test_d =  pd.read_pickle(datafolder+site_d+'/X_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    y_test_d =  pd.read_pickle(datafolder+site_d+'/y_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')

    X_train_m = X_train_m.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
    X_test_m = X_test_m.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
    X_train_d = X_train_d.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
    X_test_d = X_test_d.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
    
   #Generate cross site data     
    common_features = [x for x in X_test_d.columns if x in X_train_m.columns]

    X_train2_d = X_train_d[common_features]
    X_test2_d = X_test_d[common_features]

    X_train2_m = X_train_m.iloc[0:1]
    X_test2_m = X_test_m.iloc[0:1]

    X_train3_d = pd.concat([X_train2_m, X_train2_d]).iloc[1:]
    X_test3_d = pd.concat([X_test2_m, X_test2_d]).iloc[1:]

    X_train3_d.loc[:,X_train2_m.dtypes==bool] = X_train3_d.loc[:,X_train2_m.dtypes==bool].fillna(False)
    X_test3_d.loc[:,X_test2_m.dtypes==bool] = X_test3_d.loc[:,X_test2_m.dtypes==bool].fillna(False)        

    X_train = X_train3_d
    X_test = X_test3_d    
    y_train = y_train_d
    y_test = y_test_d    
    
    # Get AUC
#    pred = model.get_booster().predict(dtest, pred_contribs=False)
#    pred = model.predict(X_test)    
#    roc = roc_auc_score(y_test, pred)    

    pred = model.predict_proba(X_test)
    roc = roc_auc_score(y_test, pred[:,1])       
    
    shapX = pd.concat([X_train, X_test])
    shapy = pd.concat([y_train, y_test])
    
    # Calculate SHAP value
    if type(model) == xgboost.sklearn.XGBClassifier:
        dshap  = xgb.DMatrix(shapX, label=shapy)
        shap = model.get_booster().predict(dshap, pred_contribs=True)
        # Get feature importance
        model_data = pd.concat([pd.DataFrame(model.get_booster().get_score(importance_type='cover'), index=['Cover']), \
        pd.DataFrame(model.get_booster().get_score(importance_type='gain'), index=['Gain']), \
        pd.DataFrame(model.get_booster().get_score(importance_type='weight'), index=['Frequency'])]).transpose() >> mutate(Feature = X.index)
        model_data['rank'] = model_data['Gain'].rank(method='min', ascending=False)
        used_feature = list(model.get_booster().get_score().keys())        
    elif type(model) == catboost.core.CatBoostClassifier:
        cat_features = model.get_cat_feature_indices()
        pshap = Pool(data=shapX, label=shapy, cat_features=cat_features)        
        shap = model.get_feature_importance(data=pshap, type='ShapValues')
        model_data = model.get_feature_importance(prettified=True)
        model_data['Feature'] = model_data['Feature Id']
        model_data = model_data >> select('Feature', 'Importances')
        model_data['rank'] = model_data['Importances'].rank(method='min', ascending=False)     
        used_feature = list((model_data >> mask(X.Importances!=0)).Feature)
    else:
    #Using shap package example
        import shap
        explainer = shap.Explainer(model, algorithm='permutation')
        shap_valuesX = explainer.shap_values(shapX)
        #shap.summary_plot(shap_valuesX, X_test, plot_type="bar")    
        shap = shap_valuesX    
    
    # Collect SHAP value
    def CI95(data):
        if len(data) == 1:
            return (np.nan, np.nan)
        return (np.nan, np.nan)            
#        return st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) #95% confidence interval

    shap_data = list()
    shap_data_raw = list()
    for i in range(shapX.columns.shape[0]):
        df = pd.DataFrame(list(zip(shapX.iloc[:,i], shap[:, i], abs(shap[:, i]))),columns =['Name', 'val', 'absval'])
        # Check confidence interval for one data point
        plot_data = df.groupby("Name").agg([np.mean, np.var, np.std, np.median, CI95, 'size']).reset_index()
        df.index = shapX.index

        plot_data_all = df.groupby("Name").agg([np.mean, np.var, np.std, np.median, CI95, 'size']).reset_index()
        plot_data_0= df[shapy==0].groupby("Name").agg([np.mean, np.var, np.std, np.median, CI95, 'size']).reset_index()
        plot_data_1= df[shapy==1].groupby("Name").agg([np.mean, np.var, np.std, np.median, CI95, 'size']).reset_index()

        plot_data_all.columns = [''.join(x) for x in plot_data_all.columns]
        plot_data_0.columns = [x+'_0' for x in plot_data_all.columns]
        plot_data_1.columns = [x+'_1' for x in plot_data_all.columns]
        plot_data_all = plot_data_all.drop('absvalsize', axis=1)
        plot_data_0   = plot_data_0.drop('absvalsize_0', axis=1)
        plot_data_1   = plot_data_1.drop('absvalsize_1', axis=1)
        plot_data_0 = plot_data_0.rename({'Name_0':'Name'},axis=1)
        plot_data_1 = plot_data_1.rename({'Name_1':'Name'},axis=1)
        plot_data = pd.merge(plot_data_all, plot_data_0, left_on='Name', right_on='Name', how='left')
        plot_data = pd.merge(plot_data, plot_data_1, left_on='Name', right_on='Name', how='left')        
        
        plot_data = plot_data >> mutate(Feature=shapX.columns[i])
        plot_data.columns = [''.join(x) for x in plot_data.columns]
        plot_data[['valCI95down', 'valCI95up']] = pd.DataFrame(plot_data['valCI95'].tolist(), index=plot_data.index)
        plot_data[['absvalCI95down', 'absvalCI95up']] = pd.DataFrame(plot_data['absvalCI95'].tolist(), index=plot_data.index)
        plot_data = plot_data.drop(['valCI95', 'absvalCI95'],axis=1)
        shap_data.append(plot_data.copy())        
        plot_data_raw = df >> select(X.Name, X.val) >> mutate(Feature=shapX.columns[i])        
        shap_data_raw.append(plot_data_raw.copy())
    shap_data = pd.concat(shap_data)
    shap_data_raw = pd.concat(shap_data_raw)    
#    shap_data= shap_data[shap_data['Feature'].isin(used_feature)]

    # create csv for metaregression
    shap_data = shap_data >> left_join(model_data, by='Feature')
    siteyr = site_m+'_'+site_d+'_'+model_type+'_'+fs+'_'+stg+'_'+oversample+'_'+'005'+"_"+str(year)    
    shap_data = shap_data >> mutate(siteyr=siteyr) >> rename(fval=X.Name) >> rename(mean_val=X.valmean) >> rename(se_val=X.valstd) >> rename(mean_imp = X.absvalmean) >> rename(se_imp = X.absvalstd) >> rename(var_imp = X.absvalvar) >> rename(median_val = X.valmedian) >> rename(median_imp = X.absvalmedian) >> rename(var_val = X.valvar)
    shap_data['site_m'] = site_m
    shap_data['site_d'] = site_d    
    # shap_data['year'] = year
    # shap_data['stg'] = stg
    # shap_data['fs'] = fs
    # shap_data['oversample'] = oversample
    # shap_data['model'] = model_type
    # shap_data['rmcol'] = '005'
    
    # Calculate ranking base on absolute mean value of SHAP
    rank_abs_shap_max = (shap_data >> mutate(abs_shap_max = abs(X.mean_val))).loc[:,['Feature', 'abs_shap_max']].groupby(['Feature']).agg(np.max).reset_index()
    rank_abs_shap_max['rank_abs_shap_max'] = rank_abs_shap_max['abs_shap_max'].rank(method='min', ascending=False)
    shap_data = pd.merge(shap_data, rank_abs_shap_max, left_on=['Feature'], right_on=['Feature'], how='left')

    #Calculate ranking base on SHAP min max difference and variance
    tdata = shap_data.loc[:,['Feature', 'mean_val']].groupby(['Feature']).agg([np.max,np.min,np.var]).reset_index()
    tdata.columns = ['Feature', 'maxSHAP', 'minSHAP', 'varSHAP']
    tdata = (tdata >> mutate(minmax_SHAP = X.maxSHAP-X.minSHAP))
    tdata['rank_minmax_SHAP'] = tdata['minmax_SHAP'].rank(method='min', ascending=False)
    tdata['rank_var_SHAP'] = tdata['varSHAP'].rank(method='min', ascending=False)
    shap_data = pd.merge(shap_data, tdata, left_on=['Feature'], right_on=['Feature'], how='left')    

    # add auc value
    shap_data = shap_data >> mutate(auc=roc)
    
    #sort
    shap_data = shap_data.sort_values(['rank', 'fval'])

    #calculate confusion matrix
    cdata = pd.concat([pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)], axis=0)
    cmdata = cdata.melt(id_vars='FLAG', value_vars= list(cdata.columns).remove('FLAG'))
    conmat = cmdata.groupby(['FLAG', 'variable','value']).size().reset_index()
    conmat2 = conmat.pivot(index=['variable', 'value'], columns='FLAG', values=0).fillna(0).reset_index()
    conmat2.columns = ['Feature', 'fval', 'b', 'a']    
    conmat3 = cmdata.groupby(['FLAG', 'variable']).size().reset_index()
    conmat4 = conmat3.pivot(index=['variable'], columns='FLAG', values=0).fillna(0).reset_index()
    conmat4.columns = ['Feature', 'd', 'c'] 
    conmat5 = pd.merge(conmat2, conmat4, left_on='Feature', right_on='Feature', how='left')
    conmat6 = conmat5 >> mutate(d=X.d-X.b) >> mutate(c=X.c-X.a) >> mutate(num=X.a+X.b)
    conmat6['fval'] = conmat6['fval'].astype('float64')
    shap_data = pd.merge(shap_data, conmat6, left_on=['Feature', 'fval'], right_on=['Feature', 'fval'], how='left')

    #is categorical?
    X_test =  pd.read_pickle(datafolder+site_m+ '/X_test_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
    cat_features = pd.DataFrame(list(X_test.select_dtypes('bool').columns)) >> mutate(isCategorical = True)
    cat_features.columns = ['Feature', 'isCategorical']
    shap_data = pd.merge(shap_data, cat_features, right_on='Feature', left_on='Feature', how='left')
    shap_data.loc[:,'isCategorical'] = shap_data.loc[:,'isCategorical'].fillna(False)    
    
    #Collect fval range and stats
    Xdata = pd.concat([X_train, X_test], axis=0)
    try:
        filtertable = Xdata.select_dtypes(exclude=bool).agg([np.min, np.max, np.mean,np.std],axis=0).transpose()
        filtertable = filtertable.assign(upr=filtertable['mean']+3*filtertable['std']).assign(lwr=filtertable['mean']-3*filtertable['std']).reset_index().rename({'index':'Feature', 'mean':'fval_mean', 'std':'fval_std', 'upr':'fval_upr', 'lwr':'fval_lwr', 'amax':'fval_max', 'amin':'fval_min'},axis=1)
        shap_data  = pd.merge(shap_data, filtertable, right_on='Feature', left_on='Feature', how='left')
    except:
        pass
    
    #Save shap_data 
    if returnflag:
#        pass
        return shap_data, shap_data_raw
    else:
        shap_data.to_parquet(datafolder+site_m+'/shapdata_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet')
        shap_data_raw.to_parquet(datafolder+site_m+'/shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet')
    #model.to_pickle(datafolder+site+'/model_data_'+site+'_'+str(year)+'.pkl')

    toc = time.perf_counter()
    print(f"{site_m}/{site_d}:{year} finished in {toc - tic:0.4f} seconds")  
    print('Finished collectSHAP_cross_sub '+model_type+' on site '+site_m+'/'+site_d+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)

#print('done')


def collectSHAP_cross_sub_validate(configs_variable_m, configs_variable_d, fold, returnflag=False):
    '''
    This function apply internal and external data to each cross validate model of site_m
    if site_d = site_m, the validaton set is used
    else all data from site_d is used
    '''

    year=3000
    site_m, datafolder, home_directory = utils_function.get_commons(configs_variable_m)
    site_d, datafolder, home_directory = utils_function.get_commons(configs_variable_d)
    
    datafolder = configs_variable_m['datafolder']
    stg = configs_variable_m['stg']
    fs = configs_variable_m['fs']
    oversample = configs_variable_m['oversample']
    model_type = configs_variable_m['model_type']   
    drop_correlation_catboost = configs_variable_m['drop_correlation_catboost']
    
    if drop_correlation_catboost:
        suffix = 'nc'
    else:
        suffix = ''     

    if not configs_variable_m['rerun_flag'] and os.path.exists(datafolder+site_m+'/shapdata_cv_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'_'+str(fold)+'.pkl'):
        print('Existed: shapdata_cv_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'_'+str(fold)+'.pkl')
        return          
        
    
    tic = time.perf_counter()     
    print('Running collectSHAP_cross_sub '+model_type+' on site '+site_m+'/'+site_d+":"+str(year)+":"+stg+":"+fs+":"+oversample+':'+str(fold), flush = True)

    #load model
    model = pickle.load(open(datafolder+site_m+'/boosttrap_'+model_type+'_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'_'+str(fold)+'.pkl', 'rb'))

    if site_m == site_d:
        #load tables
        X_test_m = pd.read_pickle(datafolder+site_m+'/X_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+str(fold)+'.pkl')
        y_test_m = pd.read_pickle(datafolder+site_m+'/y_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+str(fold)+'.pkl')

        X_test_d =  pd.read_pickle(datafolder+site_d+'/X_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+str(fold)+'.pkl')
        y_test_d =  pd.read_pickle(datafolder+site_d+'/y_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+str(fold)+'.pkl')

    else:
        #load tables
        X_test_m = pd.read_pickle(datafolder+site_m+'/X_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+str(fold)+'.pkl')
        y_test_m = pd.read_pickle(datafolder+site_m+'/y_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+str(fold)+'.pkl')

        X_test_d =  pd.read_pickle(datafolder+site_d+'/X_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        y_test_d =  pd.read_pickle(datafolder+site_d+'/y_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')        
        
#         X_test_d1 =  pd.read_pickle(datafolder+site_d+'/X_train_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
#         y_test_d1 =  pd.read_pickle(datafolder+site_d+'/y_train_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')

#         X_test_d2 =  pd.read_pickle(datafolder+site_d+'/X_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
#         y_test_d2 =  pd.read_pickle(datafolder+site_d+'/y_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        
#         X_test_d = pd.concat([X_test_d1, X_test_d2])
#         y_test_d = pd.concat([y_test_d1, y_test_d2])        


    X_test_m = X_test_m.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
    X_test_d = X_test_d.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
            
    model = model[-3]

    #Generate cross site data     
    common_features = [x for x in X_test_d.columns if x in X_test_m.columns]

    X_test2_d = X_test_d[common_features]
    X_test2_m = X_test_m.iloc[0:1]
    X_test3_d = pd.concat([X_test2_m, X_test2_d]).iloc[1:]
    X_test3_d.loc[:,X_test2_m.dtypes==bool] = X_test3_d.loc[:,X_test2_m.dtypes==bool].fillna(False)        

    X_test = X_test3_d    
    y_test = y_test_d    

    for unmatch in X_test.dtypes[X_test.dtypes != X_test_m.dtypes].keys():
        if X_test_m.dtypes[unmatch] == bool:
            X_test[unmatch] = False
        else:
            X_test[unmatch] = np.nan        

    pred = model.predict_proba(X_test)
    roc = roc_auc_score(y_test, pred[:,1]) 

    precision, recall, thresholds = precision_recall_curve(y_test, pred[:,1])
    prauc = auc(recall, precision)     
    
    shap_data = dict()
    shap_data['site_m'] = site_m
    shap_data['site_d'] = site_d    
    # shap_data['year'] = year
    # shap_data['stg'] = stg
    # shap_data['fs'] = fs
    # shap_data['oversample'] = oversample
    # shap_data['model'] = model_type
    # shap_data['rmcol'] = '005'
    shap_data['fold'] = fold
    shap_data['roc'] = roc
    shap_data['prauc'] = prauc
    
    # shap_data['y_test'] = [np.array(y_test)]
    # shap_data['pred'] = [np.array(pred[:,1])]    

    shap_data = pd.DataFrame(shap_data,index=[1])
    shap_data.to_pickle(datafolder+site_m+'/shapdata_cv_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'_'+str(fold)+'.pkl')

    toc = time.perf_counter()
    print(f"{site_m}/{site_d}:{year} finished in {toc - tic:0.4f} seconds")  
    print('Finished collectSHAP_cross_sub '+model_type+' on site '+site_m+'/'+site_d+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
    
    
def collectSHAPraw_cross_sub(configs_variable_m, configs_variable_d, returnflag=False):

        '''
        This function apply internal and external data to each cross validate model of site_m
        if site_d = site_m, the validaton set is used
        else all data from site_d is used
        '''

        year=3000
        site_m, datafolder, home_directory = utils_function.get_commons(configs_variable_m)
        site_d, datafolder, home_directory = utils_function.get_commons(configs_variable_d)

        datafolder = configs_variable_m['datafolder']
        stg = configs_variable_m['stg']
        fs = configs_variable_m['fs']
        oversample = configs_variable_m['oversample']
        model_type = configs_variable_m['model_type']   

        drop_correlation_catboost = configs_variable_m['drop_correlation_catboost']
        if drop_correlation_catboost:
            suffix = 'nc'
        else:
            suffix = ''     

        if not configs_variable_m['rerun_flag'] and os.path.exists(datafolder+site_m+'/shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet'):
            print('Existed: shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet')
            return


        print('Running collectSHAPraw_cross_sub '+model_type+' on site '+site_m+'/'+site_d+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)
        tic = time.perf_counter()     

        #load model
        model = pickle.load(open(datafolder+site_m+'/model_'+model_type+'_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl', 'rb'))

        #load tables
        X_train_m = pd.read_pickle(datafolder+site_m+'/X_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        y_train_m = pd.read_pickle(datafolder+site_m+'/y_train_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        X_test_m = pd.read_pickle(datafolder+site_m+'/X_test_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        y_test_m = pd.read_pickle(datafolder+site_m+'/y_test_'+site_m+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')

        X_train_d = pd.read_pickle(datafolder+site_d+'/X_train_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        y_train_d = pd.read_pickle(datafolder+site_d+'/y_train_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        X_test_d =  pd.read_pickle(datafolder+site_d+'/X_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')
        y_test_d =  pd.read_pickle(datafolder+site_d+'/y_test_' +site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.pkl')

        X_train_m.index = X_train_m['PATID'] + '_' +  X_train_m['ENCOUNTERID']
        X_test_m.index = X_test_m['PATID'] + '_' +  X_test_m['ENCOUNTERID']
        X_train_d.index = X_train_d['PATID'] + '_' +  X_train_d['ENCOUNTERID']
        X_test_d.index = X_test_d['PATID'] + '_' +  X_test_d['ENCOUNTERID']

        y_train_m.index = X_train_m['PATID'] + '_' +  X_train_m['ENCOUNTERID']
        y_test_m.index = X_test_m['PATID'] + '_' +  X_test_m['ENCOUNTERID']
        y_train_d.index = X_train_d['PATID'] + '_' +  X_train_d['ENCOUNTERID']
        y_test_d.index = X_test_d['PATID'] + '_' +  X_test_d['ENCOUNTERID']
        
        X_train_m = X_train_m.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
        X_test_m = X_test_m.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
        X_train_d = X_train_d.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)
        X_test_d = X_test_d.drop(['SINCE_ADMIT', 'PATID', 'ENCOUNTERID'],axis=1)

       #Generate cross site data     
        common_features = [x for x in X_test_d.columns if x in X_train_m.columns]

        X_train2_d = X_train_d[common_features]
        X_test2_d = X_test_d[common_features]

        X_train2_m = X_train_m.iloc[0:1]
        X_test2_m = X_test_m.iloc[0:1]

        X_train3_d = pd.concat([X_train2_m, X_train2_d]).iloc[1:]
        X_test3_d = pd.concat([X_test2_m, X_test2_d]).iloc[1:]

        X_train3_d.loc[:,X_train2_m.dtypes==bool] = X_train3_d.loc[:,X_train2_m.dtypes==bool].fillna(False)
        X_test3_d.loc[:,X_test2_m.dtypes==bool] = X_test3_d.loc[:,X_test2_m.dtypes==bool].fillna(False)        

        X_train = X_train3_d
        X_test = X_test3_d    
        y_train = y_train_d
        y_test = y_test_d    

        # Get AUC
    #    pred = model.get_booster().predict(dtest, pred_contribs=False)
    #    pred = model.predict(X_test)    
    #    roc = roc_auc_score(y_test, pred)    

#         pred = model.predict_proba(X_test)
#         roc = roc_auc_score(y_test, pred[:,1])       
        
#         precision, recall, thresholds = precision_recall_curve(y_test, pred[:,1])
#         prauc = auc(recall, precision) 
        
        shapX = pd.concat([X_train, X_test])
        shapy = pd.concat([y_train, y_test])

#         # Calculashap_long[shap_long['feature']=='base_value'].to_parquet(datafolder+site_m+'/shapbase_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet')te SHAP value
        if type(model) == xgboost.sklearn.XGBClassifier:
            dshap  = xgb.DMatrix(shapX, label=shapy)
            shap = model.get_booster().predict(dshap, pred_contribs=True)
            # Get feature importance
            model_data = pd.concat([pd.DataFrame(model.get_booster().get_score(importance_type='cover'), index=['Cover']), \
            pd.DataFrame(model.get_booster().get_score(importance_type='gain'), index=['Gain']), \
            pd.DataFrame(model.get_booster().get_score(importance_type='weight'), index=['Frequency'])]).transpose() >> mutate(Feature = X.index)
            model_data['rank'] = model_data['Gain'].rank(method='min', ascending=False)
            used_feature = list(model.get_booster().get_score().keys())        
        elif type(model) == catboost.core.CatBoostClassifier:
            cat_features = model.get_cat_feature_indices()
            pshap = Pool(data=shapX, label=shapy, cat_features=cat_features)        
            shap = model.get_feature_importance(data=pshap, type='ShapValues')
            model_data = model.get_feature_importance(prettified=True)
            model_data['Feature'] = model_data['Feature Id']
            model_data = model_data >> select('Feature', 'Importances')
            model_data['rank'] = model_data['Importances'].rank(method='min', ascending=False)     
            used_feature = list((model_data >> mask(X.Importances!=0)).Feature)
        else:
        #Using shap package example
            import shap
            explainer = shap.Explainer(model, algorithm='permutation')
            shap_valuesX = explainer.shap_values(shapX)
            #shap.summary_plot(shap_valuesX, X_test, plot_type="bar")    
            shap = shap_valuesX    

        shap = pd.DataFrame(shap)
        shap.columns = list(shapX.columns)+['base_value']
        shap.index = shapX.index

        #Save shap_data 
        if returnflag:
            return shap
        else:
#            shap_long[shap_long['feature']=='base_value'].to_parquet(datafolder+site_m+'/shapbase_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet')
#            shap_final.to_parquet(datafolder+site_m+'/shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet')
            shapX.to_parquet(datafolder+site_m+'/shapdatarawX_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet')
            shap.to_parquet(datafolder+site_m+'/shapdataraw_'+model_type+'_'+site_m+'_'+site_d+'_'+str(year)+'_'+stg+'_'+fs+'_'+oversample+suffix+'.parquet')

        toc = time.perf_counter()
        print(f"{site_m}/{site_d}:{year} finished in {toc - tic:0.4f} seconds")  
        print('Finished collectSHAPraw_cross_sub '+model_type+' on site '+site_m+'/'+site_d+":"+str(year)+":"+stg+":"+fs+":"+oversample, flush = True)

    #print('done')