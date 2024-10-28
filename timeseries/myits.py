import pandas as pd
import numpy as np
import itertools
from scipy.stats import fisher_exact
import shelve
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import BSpline, make_interp_spline, interp1d
import csv
from dfply import *
import itertools
import os
import logging
from sys import getsizeof
import sklearn
import time
from sklearn.metrics import roc_auc_score
from catboost import Pool, cv
import xgboost
import catboost
import scipy.stats as st

import importlib
from joblib import Parallel, delayed
from joblib import parallel_backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, pacf
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
warnings.filterwarnings("ignore")
import scipy
import datetime
import pyEDM
from scipy import interpolate
from scipy.stats import pearsonr
from scipy.linalg import lstsq
from joblib import Parallel, delayed
from scipy import optimize

from IPython.display import HTML
from itertools import combinations, product

import pandas as pd
import numpy as np
import itertools
from scipy.stats import fisher_exact
import shelve
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
#import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import BSpline, make_interp_spline, interp1d
import csv
from dfply import *
import itertools
import os
import logging
from sys import getsizeof
import sklearn
import time
from sklearn.metrics import roc_auc_score
from catboost import Pool, cv
import xgboost
import catboost
import scipy.stats as st

import importlib
from joblib import Parallel, delayed
from joblib import parallel_backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, pacf
from statsmodels.graphics.tsaplots import plot_pacf
import warnings
warnings.filterwarnings("ignore")
import scipy
import datetime

from joblib import Parallel, delayed

import os
import statsmodels

from sklearn.linear_model import LinearRegression

import forestplot as fp
import pingouin as pg

class myits:
    
    def __init__(self):
        self.loinc1 = None
        self.loinc2 = None    
        self.site = None
        self.labxx = None
        self.data_count = None
        self.raws = None
        self.rawr = None
        self.lab = None
        self.onset = None
        self.data_count = None
        self.raws = None
        self.rawr = None
        self.tentmap_df = None
        self.E1 = None
        self.E2 = None
        self.tau1 = None
        self.tau2 = None
        self.r1p = None
        self.r2p = None
        self.rphodf1 = None
        self.rphodf2 = None 
        self.data_count_filtered = None
        self.r1s = None
        self.r2s = None
        self.r1sp = None
        self.r2sp = None 
        self.y1f = None
        self.y2f = None
        self.scale = None
        self.diff_num1 = 0
        self.diff_num2 = 0
        self.PATID = None
        self.ENCOUNTERID = None
        self.amed = None
        self.datafolder = '/home/hoyinchan/blue/Data/data2022/'
        
    def copy(self, extobj):
        self.loinc1 = extobj.loinc1
        self.loinc2 = extobj.loinc2
        self.site = extobj.site
        self.labX = extobj.labX
        self.labxx = extobj.labxx
        self.data_count = extobj.data_count
        self.raws = extobj.raws
        self.rawr = extobj.rawr
        self.lab = extobj.lab
        self.onset = extobj.onset
        self.data_count = extobj.data_count
        self.data_count_filtered = extobj.data_count_filtered
        self.raws = extobj.raws
        self.rawr = extobj.rawr
        self.tentmap_df = extobj.tentmap_df
        self.E1 = extobj.E1
        self.E2 = extobj.E2
        self.tau1 = extobj.tau1
        self.tau2 = extobj.tau2
        self.r1p = extobj.r1p
        self.r2p = extobj.r2p
        self.rphodf1 = extobj.rphodf1
        self.rphodf2 = extobj.rphodf2  
        self.amed = extobj.amed

    def get_data_2021(self, site='UTHSCSA'):
        self.site = site
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        self.lab = pd.read_csv(self.datafolder+site+'/raw/AKI_LAB.csv')
        self.lab['SPECIMEN_DATE'] = pd.to_datetime(self.lab['SPECIMEN_DATE'])

        x =  self.onset[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET', 'AKI2_ONSET', 'AKI3_ONSET']]
        x = x.bfill(axis=1)
        x = x[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET']]
        x['AKI1_ONSET'] = pd.to_datetime(x['AKI1_ONSET'])        

#        xxx = self.labxx.merge(x, on=['PATID','ENCOUNTERID'], how='left').dropna()
        xxx = self.lab.merge(x, on=['PATID','ENCOUNTERID'], how='left')
#        xxx = xxx[xxx['ADMIT_DATE'].notnull()]

        self.labX = xxx[xxx['AKI1_ONSET']-xxx['SPECIMEN_DATE']>=pd.Timedelta(1, "d")]
#        self.labxx = xxx[xxx['AKI1_ONSET']-xxx['SPECIMEN_DATE']>=pd.Timedelta(0, "d")]
        self.labX = self.labX[self.labX['SPECIMEN_DATE']>=pd.to_datetime(self.labX['ADMIT_DATE'])]
   
    def get_onsetdata_2021(self, site='UTHSCSA'):
        self.site = site
        
        self.onset = pd.read_csv(self.datafolder+site+'/raw/AKI_ONSETS.csv') 
#        self.onset = onset[onset['NONAKI_SINCE_ADMIT'].isnull()]        
        
    def get_data(self, site='UTHSCSA'):
        self.site = site
        
        self.lab = pd.read_parquet(self.datafolder+site+'/p0_lab_g_'+site+'.parquet') 
        self.lab['PATID'] = self.lab['PATID'].astype(str)
        self.lab['ENCOUNTERID'] = self.lab['ENCOUNTERID'].astype(str)
        
        x =  self.onset[['PATID','ENCOUNTERID', 'ADMIT_DATE', 'ONSET_DATE']]
        x.columns = ['PATID','ENCOUNTERID', 'ADMIT_DATE', 'AKI1_ONSET']

        xxx = self.lab.merge(x, on=['PATID','ENCOUNTERID'], how='left')

        self.labX = xxx[xxx['AKI1_ONSET']-xxx['SPECIMEN_DATE']>=pd.Timedelta(1, "d")]
        self.labX = self.labX[self.labX['SPECIMEN_DATE']>=pd.to_datetime(self.labX['ADMIT_DATE'])]
   
    def get_onsetdata(self, site='UTHSCSA'):
        self.site = site
        
        self.onset = pd.read_parquet(self.datafolder+site+'/p0_onset_'+site+'.parquet') 
        self.onset = self.onset.drop_duplicates()
        self.covid = pd.read_parquet(self.datafolder+site+'/p0_covid_status_'+site+'.parquet')
        self.covid = self.covid.drop_duplicates()

        self.onset['PATID'] = self.onset['PATID'].astype(str)
        self.onset['ENCOUNTERID'] = self.onset['ENCOUNTERID'].astype(str)
        self.covid['PATID'] = self.covid['PATID'].astype(str)
        self.covid['ENCOUNTERID'] = self.covid['ENCOUNTERID'].astype(str)
        self.onset = self.covid[self.covid['BCCOVID']][['PATID', 'ENCOUNTERID']].merge(self.onset, on = ['PATID', 'ENCOUNTERID'], how='left')

    def get_meddata_old(self, site='UTHSCSA'):
        self.site = site
        datafolder = '/home/hoyinchan/blue/Data/data2021raw/'
        datacols = ["PATID", "ENCOUNTERID", "DAYS_SINCE_ADMIT", 
                      "MEDADMIN_TYPE", "MEDADMIN_CODE", "MEDADMIN_ROUTE", "MEDADMIN_START_DATE"]
        datadtypes =  {"PATID": 'object', "ENCOUNTERID": 'object', "AKI1_SINCE_ADMIT": 'Int64', 
                         "MEDADMIN_TYPE": "object", "MEDADMIN_CODE": 'object', "MEDADMIN_ROUTE": 'object', "MEDADMIN_START_DATE": 'object'}
        self.amed = pd.read_csv(self.datafolder+site+'/raw/'+'AKI_AMED.csv', usecols=datacols, dtype=(datadtypes))
        self.amed['MEDADMIN_START_DATE'] = pd.to_datetime(self.amed['MEDADMIN_START_DATE'])
        
    def get_meddata_2021(self, site='UTHSCSA'):
        self.site = site
        import prepossess0_timeseries
        importlib.reload(prepossess0_timeseries)
        ppeng = prepossess0_timeseries.prepossess0_timeseries(site=site)
        self.amed = pd.read_parquet(self.datafolder+site+'/p0_amed_'+site+'.parquet') 
        self.amed['MEDADMIN_START_DATE'] = pd.to_datetime(self.amed['MEDADMIN_START_DATE'])

    def get_meddata(self, site='UTHSCSA'):
        self.site = site
        
        self.amed = pd.read_parquet(self.datafolder+site+'/p0_amed_'+site+'.parquet') 
                
        
    def set_loincs_pair(self):
        # self.dict_lab = {'2823-3':'potassium',
        #                 '17861-6':'calcium',
        #                 '2951-2':'sodium', 
        #                 '2160-0':'sCr',
        #                 'FLAG':'FLAG'}
        self.dict_lab = {'LG49936-4':'potassium',
                        'LG49864-8':'calcium',
                        'LG11363-5':'sodium', 
                        'LG6657-3':'sCr',
                        'FLAG':'FLAG'}
                
        self.dict_med = {'A07DA': 'DiphenoxylateLoperamide(LK)',
                          'C03CA': 'FurosemideBumetanideTorsemide(LKLNa)',
                          'C03CB': 'FurosemideBumetanideTorsemide(LKLNa)',
                          'C03EB': 'FurosemideBumetanideTorsemide(LKLNa)',
                          'C03AA': 'hydrochlorothiazide(LNa)',
                          'C03AB': 'hydrochlorothiazide(LNa)',
                          'C03AH': 'chlorothiazide(LNa)',
                          'C03BA': 'chlorthalidone(LNa)',
                          'C03EA': 'hydrochlorothiazide(LNa)',
                          'C09DX': 'SacubitrilValsartan(HK)',
                          'C03DA': 'eplerenone(HK)',
                          'C03XA': 'tolvaptan(LNa)',
                          'C09XA': 'hydrochlorothiazide(LNa)',
                          'C09DX': 'hydrochlorothiazide(LNa)',
                          'C03AX': 'hydrochlorothiazide(LNa)',
                          'C09BX': 'hydrochlorothiazide(LNa)',
                          'M05BA': 'PamidronateZoledronate(HCa)',
                          'M05BB': 'etidronate(HCa)'}          
        
        # self.loinc1 = ['2823-3', '17861-6', '2951-2']
        # self.loinc2 = ['2160-0']          
        self.loinc1 = ['LG49936-4', 'LG49864-8', 'LG11363-5']
        self.loinc2 = ['LG6657-3']          
        
        self.meds = list(self.dict_med.keys())

    def set_loincs_pair(self):
        self.dict_lab = {'LG49936-4':'potassium',
                        'LG49864-8':'calcium',
                        'LG11363-5':'sodium', 
                        'LG6657-3':'sCr',
                        'FLAG':'FLAG'}
        
        self.dict_med = {'A07DA': 'DiphenoxylateLoperamide(LK)',
                          'C03CA': 'FurosemideBumetanideTorsemide(LKLNa)',
                          'C03CB': 'FurosemideBumetanideTorsemide(LKLNa)',
                          'C03EB': 'FurosemideBumetanideTorsemide(LKLNa)',
                          'C03AA': 'hydrochlorothiazide(LNa)',
                          'C03AB': 'hydrochlorothiazide(LNa)',
                          'C03AH': 'chlorothiazide(LNa)',
                          'C03BA': 'chlorthalidone(LNa)',
                          'C03EA': 'hydrochlorothiazide(LNa)',
                          'C09DX': 'SacubitrilValsartan(HK)',
                          'C03DA': 'eplerenone(HK)',
                          'C03XA': 'tolvaptan(LNa)',
                          'C09XA': 'hydrochlorothiazide(LNa)',
                          'C09DX': 'hydrochlorothiazide(LNa)',
                          'C03AX': 'hydrochlorothiazide(LNa)',
                          'C09BX': 'hydrochlorothiazide(LNa)',
                          'M05BA': 'PamidronateZoledronate(HCa)',
                          'M05BB': 'etidronate(HCa)'}          
        
        self.loinc1 = ['LG11363-5', 'LG49864-8', 'LG49936-4']
        self.loinc2 = ['LG6657-3']          
        self.meds = list(self.dict_med.keys())        
        
        
    def extract_time_series_pair(self):
        labx = self.labX[(self.labX['LAB_LOINC'].isin(self.loinc1)) | (self.labX['LAB_LOINC'].isin(self.loinc2))]
        labx = labx[['PATID','ENCOUNTERID','SPECIMEN_DATE','LAB_LOINC','RESULT_NUM']]
        labx = labx.groupby(['PATID','ENCOUNTERID','SPECIMEN_DATE','LAB_LOINC']).mean().reset_index()
        self.labxx = labx.pivot(index=['PATID','ENCOUNTERID','SPECIMEN_DATE'], columns = 'LAB_LOINC', values='RESULT_NUM').reset_index()
        
        self.data_count = self.labxx.groupby(['PATID','ENCOUNTERID']).count().sort_values('SPECIMEN_DATE').reset_index()
        self.data_count_filtered = self.data_count[(self.data_count[self.loinc1] >=30) & (self.data_count[self.loinc2] >=30)].sort_values(self.loinc1)

    def med_code_transform(self):
        amed = self.amed
        
        # ndc -> rxnorm
        amed_rx = amed.loc[amed['MEDADMIN_TYPE'] == "RX"]
        amed_ndc = amed.loc[amed['MEDADMIN_TYPE'] == "ND"]    
        if not amed_ndc.empty:    
            ndc2rx = pd.read_parquet(self.datafolder+'med_unified_conversion_nd2rx.parquet') >> rename(MEDADMIN_CODE=X.ND)
            amed_ndc = amed_ndc >> left_join(ndc2rx, by='MEDADMIN_CODE')
            amed_ndc['MEDADMIN_TYPE'] = amed_ndc['MEDADMIN_TYPE'].where(amed_ndc['RX'].isnull(), 'RX')
            amed_ndc['MEDADMIN_CODE'] = amed_ndc['MEDADMIN_CODE'].where(amed_ndc['RX'].isnull(), amed_ndc['RX'])

        # Recombine and reseperate
        amed = pd.concat([amed_rx, amed_ndc], axis=0, ignore_index=True)
        amed_rx = amed.loc[amed['MEDADMIN_TYPE'] == "RX"]
        amed_ndc = amed.loc[amed['MEDADMIN_TYPE'] == "ND"] 

        # rxnorm -> atc
        if not amed_rx.empty:
            rxcui2atc_dtypes =  {"Rxcui": 'object', "ATC4th": 'object'}    
            rxcui2atc = pd.read_parquet(self.datafolder+'med_unified_conversion_rx2atc.parquet') >> rename(MEDADMIN_CODE=X.RX)
            amed_rx = amed_rx >> left_join(rxcui2atc, by='MEDADMIN_CODE')
            amed_rx['MEDADMIN_TYPE'] = amed_rx['MEDADMIN_TYPE'].where(amed_rx['ATC'].isnull(), 'ATC')
            amed_rx['MEDADMIN_CODE'] = amed_rx['MEDADMIN_CODE'].where(amed_rx['ATC'].isnull(), amed_rx['ATC'])   

        # Recombine and reseperate
        amed = pd.concat([amed_rx, amed_ndc], axis=0, ignore_index=True)
        
        amed['MEDADMIN_CODE_NEW'] = amed['MEDADMIN_CODE']
        amed = amed[['PATID', 'ENCOUNTERID', 'MEDADMIN_CODE_NEW', 'DAYS_SINCE_ADMIT', 'MEDADMIN_TYPE', 'MEDADMIN_CODE', 'MEDADMIN_START_DATE']]
        #amed.columns = ['PATID', 'ENCOUNTERID', 'MEDADMIN_CODE_NEW', 'DAYS_SINCE_ADMIT', 'MEDADMIN_TYPE', 'MEDADMIN_CODE']

        self.amed = amed.drop_duplicates()
        # self.amed = self.amed.merge(amed, left_on = ['PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT', 'MEDADMIN_TYPE', 'MEDADMIN_CODE'], right_on = ['PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT', 'MEDADMIN_TYPE', 'MEDADMIN_CODE'], how='inner').drop_duplicates()        

    def extract_med_data(self):
        self.amed = self.amed[self.amed['MEDADMIN_CODE_NEW'].str.contains('|'.join(self.meds))]
        self.amed = self.amed.sort_values('MEDADMIN_START_DATE').groupby(['PATID', 'ENCOUNTERID', 'MEDADMIN_CODE_NEW']).first().reset_index()

    def extract_one_med_data(self, one_med):
        pass
        
    def select_paitient(self, pos=-1):
        self.pos = pos
        tmp = self.labxx[self.labxx['ENCOUNTERID']==self.data_count_filtered['ENCOUNTERID'].iloc[pos]]
        self.PATID = tmp.iloc[0,:]['PATID']
        self.ENCOUNTERID = str(tmp.iloc[0,:]['ENCOUNTERID'])
        tmp = tmp.sort_values('SPECIMEN_DATE')
        tmp = tmp.reset_index(drop=True)
        fvi = max(tmp[self.loinc1].first_valid_index(), tmp[self.loinc2].first_valid_index())
        lvi = min(tmp[self.loinc1].last_valid_index(), tmp[self.loinc2].last_valid_index())
        tmp = tmp[fvi:lvi+1]
        tmp.index=tmp['SPECIMEN_DATE']
        self.raws = tmp[self.loinc1]
        self.rawr = tmp[self.loinc2] 
        
    def calculate_pretreatment(self):
        self.labX = self.labX[self.labX['LAB_LOINC'].isin(self.loinc1+self.loinc2)]
        self.labX['PATID'] = self.labX['PATID'].astype(str)
        self.labX['ENCOUNTERID'] = self.labX['ENCOUNTERID'].astype(str)

        #only paitients with medical record
        self.labxx = self.labX.merge(self.amed.drop('DAYS_SINCE_ADMIT',axis=1), on=['PATID', 'ENCOUNTERID'], how='left')
        self.labxx = self.labxx[self.labxx['MEDADMIN_START_DATE'].notnull()]

        self.labxx['SPECIMEN_DATE_DELTA'] = self.labxx['SPECIMEN_DATE']-self.labxx['MEDADMIN_START_DATE']
#        self.labxx = self.labxx[abs(self.labxx['SPECIMEN_DATE_DELTA'])>pd.Timedelta(1, "d")]
        self.labxx['pretreatment'] = self.labxx['SPECIMEN_DATE_DELTA']<pd.Timedelta(0, "d")
        self.labxx = self.labxx[self.labxx['RESULT_NUM'].notnull()]
        
    def pre_post_count(self, minevent=6):
        xcc = self.labxx[['PATID', 'ENCOUNTERID', 'LAB_LOINC', 'MEDADMIN_CODE_NEW', 'SPECIMEN_DATE', 'pretreatment']].groupby(['PATID', 'ENCOUNTERID', 'MEDADMIN_CODE_NEW', 'LAB_LOINC', 'pretreatment']).count().reset_index()
        xcc = xcc.pivot(index=['PATID', 'ENCOUNTERID', 'LAB_LOINC', 'MEDADMIN_CODE_NEW'], columns='pretreatment', values='SPECIMEN_DATE').fillna(0).reset_index().reset_index()
        
        cols = xcc.columns.union([False, True], sort=False)
        xcc = xcc.reindex(cols, axis=1, fill_value=0)
     
        xcc.columns = ['index', 'PATID', 'ENCOUNTERID', 'LAB_LOINC', 'MEDADMIN_CODE_NEW', 'post', 'pre']
        self.event_count_raw = xcc
        self.event_count = xcc[(xcc['post']>=6)&(xcc['pre']>=6)]

    def drop_too_few(self, minevent=6):
        self.labxx = self.labxx.merge(self.event_count[['PATID', 'ENCOUNTERID', 'LAB_LOINC', 'MEDADMIN_CODE_NEW']], on=['PATID', 'ENCOUNTERID', 'LAB_LOINC', 'MEDADMIN_CODE_NEW'], how='inner')
    
    def one_paitient_data(self, mydf=None, entid=None, loinc=None, medcode=None):
        if mydf is None:
            tmpdf = self.labxx.copy()
        else:
            tmpdf = mydf.copy()
            
        if entid is not None:
            tmpdf = tmpdf[tmpdf['ENCOUNTERID']==entid]
        if loinc is not None:
            tmpdf = tmpdf[tmpdf['LAB_LOINC']==loinc]
        if medcode is not None:
            tmpdf = tmpdf[tmpdf['MEDADMIN_CODE_NEW'].str.contains(medcode)]
            
        return tmpdf
    
    # testdf = myitsX.one_paitient_data(entid='3500451', loinc='2951-2', medcode='MED:ATC:C03CA')
    # result = myitsX.its(testdf)
    # myitsX.plot_one_paitient(testdf, result[0], result[1], result[2], result[3])
    
    def its(self, testdf):
        if testdf.empty:
            return [None, None, None, None]
        # print(testdf.iloc[0,:])
        # print(testdf.shape)
        # print(testdf.drop_duplicates().groupby('pretreatment').count())
        
        reg_pre = LinearRegression().fit((testdf[testdf['pretreatment']]['SPECIMEN_DATE_DELTA'].dt.total_seconds()/60/60/24).values.reshape(-1, 1), testdf[testdf['pretreatment']]['RESULT_NUM'].values)
        reg_post = LinearRegression().fit((testdf[~testdf['pretreatment']]['SPECIMEN_DATE_DELTA'].dt.total_seconds()/60/60/24).values.reshape(-1, 1), testdf[~testdf['pretreatment']]['RESULT_NUM'].values)
        
        return [reg_pre.intercept_, reg_pre.coef_[0], reg_post.intercept_, reg_post.coef_[0]]
    
    def its_all(self):
        self.itsdf = self.labxx.groupby(['PATID','ENCOUNTERID','MEDADMIN_CODE_NEW','LAB_LOINC']).apply(self.its).reset_index()    
        self.itsdf = self.itsdf.rename({0:'result_array'},axis=1)
        self.itsdf = pd.concat([self.itsdf, self.itsdf['result_array'].apply(pd.Series)], axis = 1).drop('result_array',axis=1)
        self.itsdf = self.itsdf.rename({0:'pre_intercept',1:'pre_slope',2:'post_intercept',3:'post_slope'},axis=1)           
    
    def plot_one_paitient(self, testdf, pre_intercept, pre_slope, post_intercept, post_slope):
        xpre = np.linspace(min(testdf[testdf['pretreatment']]['SPECIMEN_DATE_DELTA'].dt.total_seconds()/60/60/24),0).reshape(-1, 1)
        ypre = pre_intercept + pre_slope*xpre
        xpost = np.linspace(0, max(testdf[~testdf['pretreatment']]['SPECIMEN_DATE_DELTA'].dt.total_seconds()/60/60/24)).reshape(-1, 1)
        ypost = post_intercept + post_slope*xpost
        plt.plot(testdf[testdf['pretreatment']]['SPECIMEN_DATE_DELTA'].dt.total_seconds()/60/60/24, testdf[testdf['pretreatment']]['RESULT_NUM'], 'bx')
        plt.plot(testdf[~testdf['pretreatment']]['SPECIMEN_DATE_DELTA'].dt.total_seconds()/60/60/24, testdf[~testdf['pretreatment']]['RESULT_NUM'], 'rx')
        plt.plot(xpre, ypre, 'b')
        plt.plot(xpost, ypost, 'r')
        plt.axvline(x=0)
        plt.plot(0, pre_intercept, 'bo')
        plt.plot(0, post_intercept, 'ro')
        
    def calculate_stats(self, inputdf = None):
        inputisnone = False        
        if inputdf is None:
            inputisnone = True
            inputdf = self.itsdf.copy()
            if 'site' in inputdf.columns:
                inputdf = inputdf.drop('site',axis=1)
        
        inputdf['diff_intercept'] = inputdf['post_intercept']-inputdf['pre_intercept']
        inputdf['diff_slope'] = inputdf['post_slope']-inputdf['pre_slope']
        def ttest(arr):     
            return scipy.stats.ttest_1samp(arr, popmean=0)[1]       
        
        def ttest2_slope(arr, mydatatype='slope'):  
            testitsstat_slope = pg.ttest(arr['post_'+mydatatype], arr['pre_'+mydatatype], paired=False)[['p-val', 'CI95%']]
            testitsstat_slope['95ci_low'] = testitsstat_slope['CI95%'].str[0]
            testitsstat_slope['95ci_high'] = testitsstat_slope['CI95%'].str[1]
            return testitsstat_slope[['p-val', '95ci_low', '95ci_high']]
        def ttest2_intercept(arr, mydatatype='intercept'):     
            testitsstat_slope = pg.ttest(arr['post_'+mydatatype], arr['pre_'+mydatatype], paired=False)[['p-val', 'CI95%']]
            testitsstat_slope['95ci_low'] = testitsstat_slope['CI95%'].str[0]
            testitsstat_slope['95ci_high'] = testitsstat_slope['CI95%'].str[1]
            return testitsstat_slope[['p-val', '95ci_low', '95ci_high']]

        testitsstat_slope = inputdf.drop(['PATID','ENCOUNTERID'],axis=1).groupby(['MEDADMIN_CODE_NEW','LAB_LOINC']).apply(ttest2_slope).reset_index().drop('level_2',axis=1)
        testitsstat_slope.columns = ['MEDADMIN_CODE_NEW_','LAB_LOINC_', 'diff_slope_ttest2', 'diff_slope_95ci_low',  'diff_slope_95ci_high']
        testitsstat_intercept = inputdf.drop(['PATID','ENCOUNTERID'],axis=1).groupby(['MEDADMIN_CODE_NEW','LAB_LOINC']).apply(ttest2_intercept).reset_index().drop('level_2',axis=1)
        testitsstat_intercept.columns = ['MEDADMIN_CODE_NEW_','LAB_LOINC_', 'diff_intercept_ttest2', 'diff_intercept_95ci_low',  'diff_intercept_95ci_high']        
        
        if inputisnone: 
            self.testitsstat = inputdf.drop(['PATID','ENCOUNTERID','pre_intercept','pre_slope','post_intercept','post_slope'],axis=1).groupby(['MEDADMIN_CODE_NEW','LAB_LOINC']).agg([np.size, np.mean, np.std, ttest]).reset_index()            
            self.testitsstat.columns = ['_'.join(col).strip() for col in self.testitsstat.columns.values]
            self.testitsstat = self.testitsstat.merge(testitsstat_intercept, on=['MEDADMIN_CODE_NEW_','LAB_LOINC_'], how='outer').merge(testitsstat_slope, on=['MEDADMIN_CODE_NEW_','LAB_LOINC_'], how='outer')
            self.testitsstat = self.testitsstat.dropna()        
            self.testitsstatsig = self.testitsstat[(self.testitsstat['diff_intercept_ttest2']<=0.05) | (self.testitsstat['diff_slope_ttest2']<=0.05)]
            
        else:
            self.testitsstat_raw = inputdf.drop(['PATID','ENCOUNTERID','pre_intercept','pre_slope','post_intercept','post_slope'],axis=1).groupby(['MEDADMIN_CODE_NEW','LAB_LOINC']).agg([np.size, np.mean, np.std, ttest]).reset_index()            
            self.testitsstat_raw.columns = ['_'.join(col).strip() for col in self.testitsstat_raw.columns.values]
            self.testitsstat_raw = self.testitsstat_raw.merge(testitsstat_intercept, on=['MEDADMIN_CODE_NEW_','LAB_LOINC_'], how='outer').merge(testitsstat_slope, on=['MEDADMIN_CODE_NEW_','LAB_LOINC_'], how='outer')
            self.testitsstat_raw = self.testitsstat_raw.dropna()        
            self.testitsstatsig_raw = self.testitsstat_raw[(self.testitsstat_raw['diff_intercept_ttest2']<=0.05) | (self.testitsstat_raw['diff_slope_ttest2']<=0.05)]
            return self.testitsstat_raw
            
    def save_results(self):
        datarange = 'full'
        self.testitsstat['site'] = self.site
        self.testitsstatsig['site'] = self.site
        self.itsdf['site'] = self.site
        self.itsdf.to_csv(self.datafolder+'myco4_corrraw_'+datarange+'_'+self.site+'.csv')        
        self.testitsstat.dropna().to_csv(self.datafolder+'myco4_corr_'+datarange+'_'+self.site+'.csv')
        np.round(self.testitsstatsig.dropna(),3).to_csv(self.datafolder+'myco4_corrsig_'+datarange+'_'+self.site+'.csv')
        
    def run(self):
        self.calculate_pretreatment()
        self.pre_post_count()
        
        if self.event_count.empty:
            datarange = 'full'
            self.event_count_raw.to_csv(self.datafolder+'myco4_corrempty_'+datarange+'_'+self.site+'.csv')
            return
        self.drop_too_few()
        self.its_all()
        self.calculate_stats()
        self.save_results()
        
    # myitsX.combine_output(prefix='myco4_corr_', outname='mlr7_cofounder')
    def combine_output(self, prefix='myco4_corr_', outname='mlr7_cofounder'):

        files = [x for x in os.listdir() if prefix in x ]

        def combine_sub(files, range_str):
            filesx = [x for x in files if range_str in x ]
            df_u = list()
            for file in filesx:
                tmp = pd.read_csv(file)
                df_u.append(tmp)
            df_u = pd.concat(df_u)
            df_u['range'] = range_str
            return df_u

#        df_u = combine_sub(files_u, 'upper')
#        df_l = combine_sub(files_u, 'lower')
        df_f = combine_sub(files, 'full')
        
#        df = pd.concat([df_u, df_l, df_f])
        df = df_f
        #df = df[['ratio', 'Unnamed: 0', 'Coef._xy', 'P>|t|_xy', 'target', 'range', 'site']]

        df.to_csv(self.datafolder+''+outname+'.csv')
        
        
    def calculate_all3(self, mydatatype='intercept'):
        dfmlr = pd.read_csv(self.datafolder+'mlr7_cofounder.csv')

        def prepro(dfmlr,mydatatype):
            dfmlr = np.round(dfmlr,3)

            dfmlr['feature1']  = dfmlr['LAB_LOINC_']
            dfmlr['feature2']  = dfmlr['MEDADMIN_CODE_NEW_']
            dfmlr['rho']       = dfmlr['diff_'+mydatatype+'_mean']
            dfmlr['target']    = mydatatype
            dfmlr['n']         = dfmlr['diff_'+mydatatype+'_size']
            dfmlr['n0']        = dfmlr['diff_'+mydatatype+'_size']
            dfmlr['pvalue']    = dfmlr['diff_'+mydatatype+'_ttest']
            dfmlr['95ci_low']  = dfmlr['diff_'+mydatatype+'_95ci_low']
            dfmlr['95ci_high'] = dfmlr['diff_'+mydatatype+'_95ci_high']       

            dfmlr = dfmlr[['feature1','feature2','rho', 'site','target','range','pvalue','95ci_low','95ci_high', 'n', 'n0']]
            return dfmlr
        
        dfmlr = prepro(dfmlr,mydatatype)

        dfmlr_raw = pd.read_csv(self.datafolder+'mlr7raw_cofounder.csv')
        dfmlr_raw = dfmlr_raw.drop('site',axis=1)
        dfmlr_raw = dfmlr_raw.drop('range',axis=1)
        dfmlr_raw = dfmlr_raw.drop('Unnamed: 0',axis=1)
        dfmlr_raw = dfmlr_raw.drop('Unnamed: 0.1',axis=1)
        tmpsmeanstat = self.calculate_stats(inputdf=dfmlr_raw)
        tmpsmeanstat['site'] = 'MEAN'
        tmpsmeanstat['range'] = 'full'
        tmpsmeanstat = prepro(tmpsmeanstat,mydatatype)

        tmps2 = pd.concat([dfmlr, tmpsmeanstat])
        tmps2['95ci_min'] = tmps2[['95ci_low','95ci_high']].min(axis=1)
        tmps2['95ci_max'] = tmps2[['95ci_low','95ci_high']].max(axis=1)
        tmps3  = tmps2[['feature1','feature2', 'target']].drop_duplicates()

        # dict_lab = {'2157-6':'creatineKinase(HK)',
        #            '1920-8':'AST(HK)',      
        #            '2532-0':'LDH(HK)',
        #            '4542-7':'Haptoglobin(HK)', 
        #            '3084-1':'uricAcid(HK)',
        #            '2823-3':'potassium',
        #            '17861-6':'calcium',
        #            '2951-2':'sodium', 
        #            '2160-0':'sCr',
        #            'FLAG':'FLAG'}

        dict_lab = {'2157-6':'creatineKinase(HK)',
                   'LG6033-7':'AST(HK)',      
                   '2532-0':'LDH(HK)',
                   'LG44861-9':'Haptoglobin(HK)', 
                   'LG49755-8':'uricAcid(HK)',
                   'LG49936-4':'potassium',
                   'LG49864-8':'calcium',
                   'LG11363-5':'sodium', 
                   'LG6657-3':'sCr',
                   'FLAG':'FLAG'}
        
                    
        dict_med =   {'MED:ATC:A07DA': 'DiphenoxylateLoperamide(LK):A07DA',
                      'MED:ATC:C03CA': 'FurosemideBumetanideTorsemide(LKLNa):C03CA',
                      'MED:ATC:C03CB': 'FurosemideBumetanideTorsemide(LKLNa):C03CA',
                      'MED:ATC:C03EB': 'FurosemideBumetanideTorsemide(LKLNa):C03EB',
                      'MED:ATC:C03AA': 'hydrochlorothiazide(LNa):C03AA',
                      'MED:ATC:C03AB': 'hydrochlorothiazide(LNa):C03AB',
                      'MED:ATC:C03AH': 'chlorothiazide(LNa):C03AH',
                      'MED:ATC:C03BA': 'chlorthalidone(LNa):C03BA',
                      'MED:ATC:C03EA': 'hydrochlorothiazide(LNa):C03EA',
                      'MED:ATC:C09DX': 'SacubitrilValsartan(HK):C09DX',
                      'MED:ATC:C03DA': 'eplerenone(HK):C03DA',
                      'MED:ATC:C03XA': 'tolvaptan(LNa):C03XA',
                      'MED:ATC:C09XA': 'hydrochlorothiazide(LNa):C09XA',
                      'MED:ATC:C09DX': 'hydrochlorothiazide(LNa):C09DX',
                      'MED:ATC:C03AX': 'hydrochlorothiazide(LNa):C03AX',
                      'MED:ATC:C09BX': 'hydrochlorothiazide(LNa):C09BX',
                      'MED:ATC:M05BA': 'PamidronateZoledronate(HCa):M05BA',
                      'MED:ATC:M05BB': 'etidronate(HCa):M05BB'}

        plotdict = dict()

        tmps3 = tmps3.replace(dict_lab)
        tmps2 = tmps2.replace(dict_lab)

        tmps3 = tmps3.replace(dict_med)
        tmps2 = tmps2.replace(dict_med)      

        plot_prefix = "plot4_"
        for row in tmps3.iterrows():
            f1 = row[1][0]
            f2 = row[1][1]
            f3 = row[1][2]
            tmps2X = tmps2[(tmps2['feature1']==f1) & (tmps2['feature2']==f2) & (tmps2['target']==f3)]
            tmps2X  = tmps2X.reset_index(drop=True)
            fp.forestplot(tmps2X, 
                      estimate="rho",  # col containing estimated effect size 
                      ll="95ci_min", hl="95ci_max",  # columns containing conf. int. lower and higher limits
                      varlabel="site",  # column containing variable label
                      ylabel="Confidence interval",  # y-label title
                      xlabel=f1+' vs '+f2+ '('+f3+')',
                      groupvar="range",
                      rightannote=["pvalue", "n", "n0"],
                      right_annoteheaders=["pvalue", "n(cofounder)", "n(base)"])
            plt.savefig(plot_prefix+f1+'_'+f2+'_'+f3+"_.png", bbox_inches="tight")

    def combineimg(self):
        #Read the two images
        def combineimg_sub(f1,f2,f3,f4,outname):
            image1 = Image.open(f1)
            image2 = Image.open(f2)
            image3 = Image.open(f3)
            image4 = Image.open(f4)            

            #resize, first image
            image2 = image2.resize(image1.size)
            image3 = image3.resize(image1.size)
            image4 = image4.resize(image1.size)

            image1_size = image1.size
            image2_size = image2.size
            image3_size = image3.size
            image4_size = image4.size

            new_image = Image.new('RGB',(2*image1_size[0], 2*image1_size[1]), (250,250,250))
            new_image.paste(image1,(0,0))
            new_image.paste(image2,(image1_size[0],0))
            new_image.paste(image3,(0,image1_size[1]))
            new_image.paste(image4,(image1_size[0],image1_size[1]))

            new_image.save(outname)

        files = [x for x in os.listdir() if 'plot4_' in x]

        filesdf = pd.DataFrame(pd.DataFrame(files)[0].str.split('.').str[0].str.split('_').tolist(), columns= ['A','B','C','E','F'])

        filesdf['fname'] = filesdf['A']+'_'+filesdf['B']+'_'+filesdf['C']+'_'+filesdf['E']+'_'+filesdf['F']+'.png'

        filesdf = filesdf[filesdf['E']=='intercept']
        filesdfxy = filesdf[filesdf['B']!='sCr'].merge(filesdf[filesdf['B']=='sCr'], on=['A','C','E','F'], how='left')

        for index, row in filesdfxy.iterrows():
            outname = row['A']+'d_'+row['B_x']+'_'+row['C']+'.png'