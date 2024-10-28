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
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
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
import math
import matplotlib.pyplot as plt
from glob import glob

from joblib import parallel_backend
from joblib import Parallel, delayed
import requests

class labunify:
    
    def __init__(self):
#        self.sites = ['MCRI', 'MCW', 'UIOWA', 'UMHC', 'UNMC', 'UofU', 'UPITT', 'UTHSCSA', 'UTSW']
#        self.datafolder = '/home/hoyinchan/blue/Data/data2021/data2021/'
        self.sites = ['KUMC', 'UTSW', 'MCW', 'UofU', 'UIOWA', 'UMHC', 'UPITT', 'UTHSCSA', 'UNMC']
        self.datafolder = '/blue/yonghui.wu/hoyinchan/Data/data2022/'

        self.home_directory = "/home/hoyinchan/code/AKI_CDM_PY/"
        self.laball = None
        self.loincmap = None
        self.loincmap2 = None
        self.loincmap3 = None
        self.drop_list1 = None
        self.bigtable = None
        self.labstdunit = None
        
    def read_lab(self):
        labs = list()
        for site in self.sites:
            lab = pd.read_pickle(self.datafolder+site+'/p0_lab_'+site+'.pkl')
            lab = lab[["PATID", "ENCOUNTERID", "RESULT_NUM", "LAB_LOINC", "LAB_PX_TYPE", "RESULT_UNIT", "RESULT_QUAL"]]
            lab['site'] = site
            labs.append(lab)
        self.laball = pd.concat(labs, axis=0, sort=False)    
        self.laball = self.laball[np.logical_or(self.laball['RESULT_NUM'].notnull(), self.laball['RESULT_QUAL']!='NI')]
        
    def read_loinc(self):
        self.loincmap  = pd.read_csv(self.home_directory+'loinc/LoincTableCore/MapTo.csv')
        self.loincmap2 =pd.read_csv(self.home_directory+'loinc/LoincTable/Loinc.csv', low_memory=False)
        self.loincmap3 =pd.read_csv(self.home_directory+'loinc/AccessoryFiles/GroupFile/GroupLoincTerms.csv') 
        
    def get_all_relation(self):
        laballuni = self.laball[["LAB_LOINC", "RESULT_UNIT", "site", "PATID", "ENCOUNTERID"]].drop_duplicates().drop("PATID",axis=1)
        laballuni2 = laballuni.groupby(["LAB_LOINC", "RESULT_UNIT", "site"]).count().reset_index()
        laballunix = laballuni2.merge(self.loincmap3, left_on='LAB_LOINC', right_on='LoincNumber', how='left')
        laballunixx = laballunix.merge(self.loincmap2[['LOINC_NUM', 'SCALE_TYP','EXAMPLE_UCUM_UNITS', 'LONG_COMMON_NAME']], left_on='LAB_LOINC',
                                       right_on='LOINC_NUM', how='left'
                                      ).drop(['Archetype', 'LoincNumber', 'LOINC_NUM', 'LongCommonName'],axis=1)
        laballunixx.GroupId.fillna(laballunixx.LAB_LOINC, inplace=True)        
        laballunixx2 = laballunixx[laballunixx['SCALE_TYP']=='Qn']
        tcount = laballunixx2[['LAB_LOINC', 'site', 'ENCOUNTERID']].groupby(['LAB_LOINC', 'site']).sum().reset_index()
        tcount.columns = ['LAB_LOINC', 'site', 'sum']
        laballunixx5 = laballunixx2.merge(tcount, on=['LAB_LOINC', 'site'], how='left')
        laballunixx5['ratio'] = laballunixx5['ENCOUNTERID']/laballunixx5['sum']
        self.drop_list1 = laballunixx5[laballunixx5['ratio']<0.05]
        self.bigtable = laballunixx5[laballunixx5['ratio']>=0.05]
        
    def get_lab_site_count(self):
        return self.bigtable[['LAB_LOINC', 'site','RESULT_UNIT']].drop_duplicates().groupby(['LAB_LOINC',
                            'site']).count().reset_index().sort_values('RESULT_UNIT')
    
    def get_consensus_unit(self):
        laballunixx5overall = self.bigtable[np.logical_not(self.bigtable['RESULT_UNIT'].str.contains('NI|OT|UN'))][['GroupId', 'RESULT_UNIT',
                                           'site']].drop_duplicates().groupby(['GroupId',
                                            'RESULT_UNIT']).count().reset_index().sort_values('RESULT_UNIT')
        self.labstdunit = laballunixx5overall.sort_values('site', ascending=False).groupby(['GroupId']).first().reset_index()
        self.labstdunit = self.labstdunit[['GroupId', 'RESULT_UNIT']]
        self.labstdunit.columns = ['GroupId', 'RESULT_UNIT_CONSENSUS']

    def copy(self, obj):
        self.sites = obj.sites
        self.datafolder = obj.datafolder
        self.home_directory = obj.home_directory
        self.laball = obj.laball
        self.loincmap = obj.loincmap
        self.loincmap2 = obj.loincmap2
        self.loincmap3 = obj.loincmap3
        self.drop_list1 = obj.drop_list1
        self.bigtable = obj.bigtable
        self.labstdunit = obj.labstdunit
        
    def fileterbyAKI(self):
        onsets = list()
        for site in self.sites:
            onset = pd.read_pickle(self.datafolder+site+'/p0_onset_'+site+'.pkl')
            onset['site'] = site
            onsets.append(onset)
        onsets = pd.concat(onsets)        
        onsetsc = onsets[np.logical_or(np.logical_or(onsets['AKI2_ONSET'].notnull(), onsets['AKI2_ONSET'].notnull()),
                                       onsets['AKI3_ONSET'].notnull())][["PATID", "ENCOUNTERID", 'site']].drop_duplicates()
        onsetsc = onsetsc.drop("PATID",axis=1).groupby('site').count().reset_index()
        onsetsc.columns = ['site', 'AKIENCOUNTER']
        self.bigtable = self.bigtable.merge(onsetsc, on='site', how='left')
        self.bigtable['AKIratio'] = self.bigtable['ENCOUNTERID']/self.bigtable['AKIENCOUNTER']
        self.drop_list2 = self.bigtable[self.bigtable['ratio']<0.05].copy()
        self.bigtable = self.bigtable[self.bigtable['AKIratio']>=0.05]
        
    def gen_local_conversion_table(self):
        tmpx = self.bigtable[['LAB_LOINC', 'site', 'RESULT_UNIT']].drop_duplicates().groupby(['LAB_LOINC', 'site']).count().reset_index()
        tmpx.columns = ['LAB_LOINC', 'site', 'RESULT_UNIT_COUNT']
        tmpy = self.bigtable.merge(tmpx[tmpx['RESULT_UNIT_COUNT']>1], on = ['LAB_LOINC', 'site'], how='right')[['LAB_LOINC', 'site', 'RESULT_UNIT', 'LONG_COMMON_NAME']].drop_duplicates().sort_values(['LAB_LOINC', 'site'])
        tmpy = tmpy.merge(tmpx, on = ['LAB_LOINC', 'site'], how='left')
        tmpyy = tmpy[tmpy['RESULT_UNIT'].str.contains('NI|OT|UN')][['LAB_LOINC', 'site', 'RESULT_UNIT']].groupby(['LAB_LOINC', 'site']).count().reset_index()
        tmpyy.columns = ['LAB_LOINC', 'site', 'RESULT_UNIT_COUNT_OT']
        tmpy2 = tmpy.merge(tmpyy, on = ['LAB_LOINC', 'site'], how='left').fillna(0)
        tmpy2 = tmpy2[tmpy2['RESULT_UNIT_COUNT_OT']>0]
        tmpy2['RESULT_UNIT_COUNT_NON_OT'] = tmpy2['RESULT_UNIT_COUNT'] - tmpy2['RESULT_UNIT_COUNT_OT']
        tmpy3 = tmpy2[tmpy2['RESULT_UNIT_COUNT_NON_OT']>0]
        tmpy4 = tmpy3.merge(self.bigtable[['LAB_LOINC', 'RESULT_UNIT', 'site', 'ENCOUNTERID', 'ratio']].drop_duplicates(), on=['LAB_LOINC','site', 'RESULT_UNIT'], how='left')
        tmpy4maxV = tmpy4[np.logical_not(tmpy4['RESULT_UNIT'].str.contains('NI|OT|UN'))][['LAB_LOINC', 'site', 'RESULT_UNIT', 'ENCOUNTERID']].sort_values('ENCOUNTERID', ascending=False).groupby(['LAB_LOINC', 'site']).first().reset_index().drop('ENCOUNTERID',axis=1)
        tmpy4maxV.columns = ['LAB_LOINC', 'site', 'RESULT_UNIT_MAX']
        tmpy4 = tmpy4.merge(tmpy4maxV, on=['LAB_LOINC','site'], how='left')
        self.local_custom_convert = tmpy4[['LAB_LOINC', 'site', 'RESULT_UNIT', 'RESULT_UNIT_MAX', 'LONG_COMMON_NAME']].copy()
        self.local_custom_convert.columns = ['LAB_LOINC', 'site', 'SOURCE_UNIT', 'TARGET_UNIT', 'LONG_COMMON_NAME']
        self.local_custom_convert['Multipliyer'] = 1
    
    def create_conversion1(self):
        file1 = open('loinc_cinversion.txt', 'r')
        Lines = file1.readlines()
        file1.close()
        i = 0
        source_loinc = dict()
        source_unit = dict()
        target_loinc = dict()
        target_unit = dict()
        while i < len(Lines):
            if 'Conversion ' in Lines[i]:
                x = Lines[i].split(' ')[1].split('\t')[0].split('\n')[0]
                i = i+2
                source_loinc[x] = Lines[i].split('\t')[1].replace('\t','')
                source_unit[x] = Lines[i].split('(')[-1].split(')')[0]
                i = i+1
                target_loinc[x] = Lines[i].split('\t')[1].replace('\t','')
                target_unit[x] = Lines[i].split('(')[-1].split(')')[0]     
            i=i+1
            
        file1 = open('UnitEquations 20170210-courier.txt', 'r')
        Lines = file1.readlines()
        file1.close()
        source_loinc2 = list()
        target_loinc2 = list()
        factor = list()
        for l in Lines:
            source_loinc2.append(l.split(']')[0].split('[')[1].replace('\t',''))
            target_loinc2.append(l.split(']')[1].split('[')[1].replace('\t',''))
            factor.append(l.split(']')[2].split('\n')[0])

        cv1 = pd.DataFrame([source_loinc, source_unit, target_loinc, target_unit]).T
        cv1.columns = ['target_loinc', 'target_unit', 'source_loinc', 'source_unit']
        cv2 = pd.DataFrame([source_loinc2, target_loinc2, factor]).T
        cv2.columns = ['source_loinc', 'target_loinc', 'factor']

        cv = cv1.merge(cv2, on=['source_loinc', 'target_loinc'], how='outer')
        cv['factor'] = '1'+cv['factor']
        
        def eval2(formula):
            if '10^' in formula:
                return formula
            elif '^-1' in formula:
                return formula
            else:
                return eval(formula)

        cv['factor2'] = cv['factor'].apply(eval2)        
        self.cv1 = cv
    
    def generate_UCUM(self):
        print('start')
        charreplace = {"%":"%25","\[":"%5B","\]":"%5D","\{":"%7B","\}":"%7D"}
        tmpx = self.bigtable[['LAB_LOINC', 'site', 'GroupId', 'EXAMPLE_UCUM_UNITS']].drop_duplicates()
        tmpyy = tmpx[['site', 'GroupId', 'EXAMPLE_UCUM_UNITS']].groupby(['GroupId',
                'EXAMPLE_UCUM_UNITS']).count().reset_index().sort_values('site', ascending=False).groupby(['GroupId']).first().reset_index().drop('site',axis=1)
        tmpyy.columns = ['GroupId', 'EXAMPLE_UCUM_UNITS_FINAL']

        tmpz = self.bigtable.merge(tmpyy, on='GroupId', how='left')
        tmpzz = tmpz[['LAB_LOINC', 'RESULT_UNIT', 'GroupId', 'EXAMPLE_UCUM_UNITS', 'EXAMPLE_UCUM_UNITS_FINAL']].drop_duplicates()
        tmpzzz = tmpzz[tmpzz['RESULT_UNIT'] != tmpzz['EXAMPLE_UCUM_UNITS_FINAL']]

        UCUMunit = tmpzzz.merge(self.labstdunit, on='GroupId', how='outer')
        UCUMunit['FINAL_UNIT'] = np.where(UCUMunit['EXAMPLE_UCUM_UNITS_FINAL'].str.contains(';'), UCUMunit['RESULT_UNIT_CONSENSUS'], UCUMunit['EXAMPLE_UCUM_UNITS_FINAL'])
        UCUMunitX = UCUMunit[np.logical_not(UCUMunit['RESULT_UNIT'].str.contains('NI|OT|UN'))]

        UCUMunitX['FINAL_Multiplyer']=np.nan
        UCUMunitX['RESULT_UNIT_API'] = UCUMunitX['RESULT_UNIT'].replace(charreplace, regex=False)
        UCUMunitX['FINAL_UNIT_API'] = UCUMunitX['FINAL_UNIT'].replace(charreplace, regex=False)

        def get_conversion(df):
            if not np.isnan(df['factor_final']):
                return df['factor_final']
            convert_dict={"%":"%25","\[":"%5B","\]":"%5D","\{":"%7B","\}":"%7D"}
            if df['RESULT_UNIT_API'] != df['FINAL_UNIT_API']:
                print(f"converting : {df['RESULT_UNIT_API']}")
                urlstring = 'https://ucum.nlm.nih.gov/ucum-service/v1/ucumtransform/1/from/'+df['RESULT_UNIT_API']+'/to/'+df['FINAL_UNIT_API']+'/LOINC/'+df['LAB_LOINC']     
        #        print(urlstring)
                response = requests.get(urlstring)
                if response.ok:
                    try:
                        factor = float(response.text.split('<ResultQuantity>')[-1].split('</ResultQuantity>')[0])
                        print(df['RESULT_UNIT_API']+df['FINAL_UNIT_API']+str(factor))                
                    except:
                        factor = np.nan
                else:
                    factor = np.nan
                time.sleep(1)
            else:
                factor = 1
            return factor
        
        UCUMunitX['FINAL_UNIT'] = UCUMunitX['FINAL_UNIT'].str.replace('\[IU\]','U').str.replace('\[arb\'U\]','U')        
        UCUMunitX['FINAL_UNIT_API'] = UCUMunitX['FINAL_UNIT_API'].str.replace('\[IU\]','U').str.replace('\[arb\'U\]','U')
        UCUMunitX['factor_final'] = np.nan
        print(f"Starting convert {UCUMunitX.shape[0]} units")        
        UCUMunitX['factor_final'] = UCUMunitX.apply(get_conversion,axis=1)
        self.UCUMunitX = UCUMunitX
    
    def handle_qualitative(self):
        laballuni = self.laball[["LAB_LOINC", "RESULT_NUM", "RESULT_QUAL", "site", "PATID", "ENCOUNTERID"]].drop_duplicates().drop("PATID",axis=1)
        laballuni2 = laballuni[laballuni['RESULT_NUM'].isnull()]
        laballuni3 = laballuni2[["LAB_LOINC", "RESULT_QUAL"]].drop_duplicates()
        laballuni4 = laballuni3.merge(self.loincmap2[self.loincmap2['SCALE_TYP']!='Qn'][['LOINC_NUM', 'SCALE_TYP', 'LONG_COMMON_NAME']], left_on='LAB_LOINC',
                                       right_on='LOINC_NUM', how='left')

        laballuni5 = laballuni4.merge(self.loincmap3, left_on='LAB_LOINC', right_on='LoincNumber', how='left')
        laballuni5.GroupId.fillna(laballuni5.LAB_LOINC, inplace=True)        

        laballuniby = laballuni5[['GroupId', 'RESULT_QUAL']].drop_duplicates().groupby('GroupId').count().reset_index()
        laballuniby.columns = ['GroupId', 'RESULT_QUAL_COUNT']
        laballuni6 = laballuni5.merge(laballuniby, on='GroupId')
#            laballuni6[laballuni6['RESULT_QUAL_COUNT']>2][['GroupId', 'RESULT_QUAL']].drop_duplicates()
        self.UCUMqualX = laballuni6[['LAB_LOINC', 'GroupId']]
    
#labobj.read_lab()
#labobj.laball = laball
#labobj.read_loinc()
#labobj.get_all_relation()
#self.get_consensus_unit()       