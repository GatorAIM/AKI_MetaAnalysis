"""
This module contains a set of function that process different PCORNET table 
Split processing into per year
Long table into wide format
One hot for boolean values
"""

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
import logging
from sys import getsizeof
import utils_function
import os
import itertools
import pickle5 as pickle

def onset_old(configs_variables, year):                
    """
    The module process the onset table
    1) Split iinto years
    2) Define the onset time as the last stage

    Input:
    p0_onset_{site}.pkl - Long format PCORNET table

    Output:
    onset_{site}_{str(year)}.pkl - vital table (cont)
    """

    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
    # load tables    
    onset = pd.read_parquet(datafolder+site+'/p0_onset_'+site+'.pkl')        
    onset['PATID'] = onset['PATID'].astype(str)
    onset['ENCOUNTERID'] = onset['ENCOUNTERID'].astype(str)
    
    
    print('Running onset on site '+site+":"+str(year), flush = True)            
    #get paitient by year
    onset.loc[:,'ADMIT_DATE'] = pd.to_datetime(onset['ADMIT_DATE'])
    onset_yr = onset.query("ADMIT_DATE >= '"+str(year)+"/01/01' and ADMIT_DATE <= '"+str(year)+"/12/31'")

    # get non-AKI paitients
    onset_yr_aki0 = onset_yr[onset_yr["NONAKI_SINCE_ADMIT"].notnull()]
    onset_yr_aki0_select = onset_yr_aki0[["PATID", "ENCOUNTERID", "NONAKI_SINCE_ADMIT"]]
    onset_yr_aki0_select = onset_yr_aki0_select.assign(FLAG = 0)
    onset_yr_aki0_select = onset_yr_aki0_select >> rename(SINCE_ADMIT=X.NONAKI_SINCE_ADMIT)

    # Get AKI1 paitients    
    onset_yr_aki1 = onset_yr[np.logical_and(onset_yr["AKI1_SINCE_ADMIT"].notnull(), np.logical_and(onset_yr["AKI2_SINCE_ADMIT"].isnull(), onset_yr["AKI3_SINCE_ADMIT"].isnull()))]
    onset_yr_aki1_select = onset_yr_aki1[["PATID", "ENCOUNTERID", "AKI1_SINCE_ADMIT"]]
    onset_yr_aki1_select = onset_yr_aki1_select.assign(FLAG = 1)
    onset_yr_aki1_select = onset_yr_aki1_select >> rename(SINCE_ADMIT=X.AKI1_SINCE_ADMIT)
    
    # Get AKI2 paitients    
    onset_yr_aki2 = onset_yr[np.logical_and(onset_yr["AKI2_SINCE_ADMIT"].notnull(), onset_yr["AKI3_SINCE_ADMIT"].isnull())]
    onset_yr_aki2_select = onset_yr_aki2[["PATID", "ENCOUNTERID", "AKI2_SINCE_ADMIT"]]
    onset_yr_aki2_select = onset_yr_aki2_select.assign(FLAG = 2)
    onset_yr_aki2_select = onset_yr_aki2_select >> rename(SINCE_ADMIT=X.AKI2_SINCE_ADMIT)    

    # Get AKI3 paitients    
    onset_yr_aki3 = onset_yr[onset_yr["AKI3_SINCE_ADMIT"].notnull()]
    onset_yr_aki3_select = onset_yr_aki3[["PATID", "ENCOUNTERID", "AKI3_SINCE_ADMIT"]]
    onset_yr_aki3_select = onset_yr_aki3_select.assign(FLAG = 3)
    onset_yr_aki3_select = onset_yr_aki3_select >> rename(SINCE_ADMIT=X.AKI3_SINCE_ADMIT)        
    
    newdf = pd.concat([onset_yr_aki1_select, onset_yr_aki0_select, onset_yr_aki2_select, onset_yr_aki3_select], axis=0, sort=False).reset_index(drop=True)

    
    # This Table has to be straightly nonnull, (sometime the SINCE_ADMIT is null)
    # onset_yr_aki0_select = onset_yr_aki0_select.dropna()
    # onset_yr_aki1_select = onset_yr_aki1_select.dropna()
    # onset_yr_aki2_select = onset_yr_aki2_select.dropna()
    # onset_yr_aki3_select = onset_yr_aki3_select.dropna()
    
    newdf = newdf.dropna()
    #Save table   
    newdf.to_pickle(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl')

    print('Finished onset on site '+site+":"+str(year), flush = True)
    
def onset(configs_variables, year):                
    '''
    The module process the onset table
    1) Split iinto years
    2) Define the onset time as the last stage

    Input:
    p0_onset_{site}.pkl - Long format PCORNET table

    Output:
    onset_{site}_{str(year)}.pkl - vital table (cont)
    '''

    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    
    
    if not configs_variables['rerun_flag'] and os.path.exists(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl'):
        print('Existed: onset_'+site+'_'+str(year)+'.pkl')
        return
    
    
    # load tables    
    onset = pd.read_parquet(datafolder+site+'/p0_onset_'+site+'.parquet')        
    onset['PATID'] = onset['PATID'].astype(str)
    onset['ENCOUNTERID'] = onset['ENCOUNTERID'].astype(str)    
    onset = onset.drop_duplicates()
    onset= onset[onset['SINCE_ADMIT']!=0]
    
    print('Running onset on site '+site+":"+str(year), flush = True)            
    #get paitient by year
    onset.loc[:,'ADMIT_DATE'] = pd.to_datetime(onset['ADMIT_DATE'])
    onset_yr = onset.query("ADMIT_DATE >= '"+str(year)+"/01/01' and ADMIT_DATE <= '"+str(year)+"/12/31'")
    
    onset_yr = onset_yr[["PATID", "ENCOUNTERID", "SINCE_ADMIT", "AKI_STAGE"]]
    onset_yr.columns = ["PATID", "ENCOUNTERID", "SINCE_ADMIT", "FLAG"]
    
    
    onset_yr.to_pickle(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl')
    print('Finished onset on site '+site+":"+str(year), flush = True)
    
def vital(configs_variables, year):
    '''
    The module process the vital table to get the last avaliable data 24 hour before onset
    1) Include only all data 1 day before onset
    2) Calculate Daily Average
    3) Drop all data before admit
    4) Collect last avaliable data

    Input:
    p0_vital_{site}.pkl - Long format PCORNET table

    Output:
    vital_{site}_{str(year)}.pkl - vital table (cont)
    '''

    site, datafolder, home_directory = utils_function.get_commons(configs_variables)    
    print('Running vital on site '+site+":"+str(year), flush = True)
    

    if not configs_variables['rerun_flag'] and os.path.exists(datafolder+site+'/vital_'+site+'_'+str(year)+'.pkl'):
        print('Existed: vital_'+site+'_'+str(year)+'.pkl')
        return    
    
    # load tables
#    vital = pd.read_parquet(datafolder+site+'/p0_vital_'+site+'.parquet')

    if configs_variables['remove_outliner_flag']:
        print('Loading remove outliner vital')
        vital = pd.read_parquet(datafolder+site+'/p0_vital_old_'+site+'_'+str(year)+'_nooutliner.parquet')        
    else:
        vital = pd.read_parquet(datafolder+site+'/p0_vital_old_'+site+'_'+str(year)+'.parquet')
    
    
    vital['PATID'] = vital['PATID'].astype(str)
    vital['ENCOUNTERID'] = vital['ENCOUNTERID'].astype(str)
    
    # Get the patient records in onset
    # Calculate 'FUTURE' column as ONSET_DAY-MEASURE_DAY
    newdfX = pd.read_pickle(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl')
    newdfX = newdfX >> select('PATID', 'ENCOUNTERID', 'SINCE_ADMIT') >> mutate(dummy = True)
    vital = (pd.merge(vital, newdfX, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left').fillna({'dummy': False}) >> mask(X.dummy) >> select(~X.dummy) >> mutate(FUTURE=X.SINCE_ADMIT-X.DAYS_SINCE_ADMIT)).reset_index(drop=True)
    
    #24 hours prediction 
    vital = vital[vital['FUTURE']>0].drop(['SINCE_ADMIT','FUTURE'],axis=1)
    #Only include in-hosipital record
    vital = vital[vital['DAYS_SINCE_ADMIT']>=0]
    
    #Calculate daily average
    vital_mean = vital.groupby(['PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT']).mean().reset_index()

    #Transform vital Table (Row over the previous value if unknown) (Continuous)
    vital_list = []
    #Vital table drop data before admit
    vital_sys = vital_mean >> select('PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT', 'SYSTOLIC')
    vital_dia = vital_mean >> select('PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT', 'DIASTOLIC')
    vital_bmi = vital_mean >> select('PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT', 'ORIGINAL_BMI')
    vital_wt = vital_mean >> select('PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT', 'WT')
   
    #get the last avaliable value                                
    vital_sys_p = vital_sys.dropna().sort_values(['PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT']).groupby(['PATID', 'ENCOUNTERID']).agg({'SYSTOLIC':'last'}).reset_index()    
    vital_dia_p = vital_dia.dropna().sort_values(['PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT']).groupby(['PATID', 'ENCOUNTERID']).agg({'DIASTOLIC':'last'}).reset_index()    
    vital_bmi_p = vital_bmi.dropna().sort_values(['PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT']).groupby(['PATID', 'ENCOUNTERID']).agg({'ORIGINAL_BMI':'last'}).reset_index()    
    vital_wt_p  =  vital_wt.dropna().sort_values(['PATID', 'ENCOUNTERID', 'DAYS_SINCE_ADMIT']).groupby(['PATID', 'ENCOUNTERID']).agg({'WT':'last'}).reset_index()    

    
    #Combine back into one vital table
    vital_t = pd.merge(vital_sys_p, vital_dia_p, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='outer')
    vital_t = pd.merge(vital_t, vital_bmi_p, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='outer')
    vital_t = pd.merge(vital_t, vital_wt_p, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='outer')        

    #Save table
    vital_t.to_pickle(datafolder+site+'/vital_'+site+'_'+str(year)+'.pkl')

    #consistency check
    if vital_t.empty:
        logging.basicConfig(filename='vital.log', filemode='a')    
        print('DATAFRAME EMPTY!!!!!! '+site+":"+str(year), flush = True)
        logging.error('vital: DATAFRAME EMPTY!!!!!! '+site+":"+str(year))
        logging.shutdown()

    print('Finished vital on site '+site+":"+str(year), flush = True)
#    return vital_t

def demo(configs_variables, year):
    '''
    The module process the demographic table
    1) Extract demographic record f patients in onset
    2) Transform 'SEX', 'RACE', 'HISPANIC' into onehot vector

    Input:
    p0_demo_{site}.pkl - Long format PCORNET table

    Output:
    demo_{site}_{str(year)}.pkl - Demo table (AGE+boolean)
    '''
    
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)    
    print('Running demo on site '+site+":"+str(year), flush = True)

    
        

    if not configs_variables['rerun_flag'] and os.path.exists(datafolder+site+'/demo_'+site+'_'+str(year)+'.pkl'):
        print('Existed: demo_'+site+'_'+str(year)+'.pkl')
        return        
    
    # load tables
    demo = pd.read_parquet(datafolder+site+'/p0_demo_'+site+'_'+str(year)+'.parquet')
    demo['PATID'] = demo['PATID'].astype(str)
    demo['ENCOUNTERID'] = demo['ENCOUNTERID'].astype(str)    
    demo['AGE'] = demo['AGE'].astype(float)
    demo = demo.drop_duplicates()
    
    # Get the patient records in onset
    newdfX = pd.read_pickle(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl')
    newdfX = newdfX >> select('PATID', 'ENCOUNTERID') >> mutate(dummy = True) >> distinct()
    demo = (pd.merge(demo, newdfX, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left').fillna({'dummy': False}) >> mask(X.dummy) >> select(~X.dummy)).reset_index(drop=True)

    #onehot transform demo 
    var = ['SEX', 'RACE', 'HISPANIC']
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(demo[var])
    demo_onehot_cat = pd.DataFrame(enc.transform(demo[var]).toarray(), columns=enc.get_feature_names(var)).astype('bool')
    demo_one = pd.concat([demo[['PATID', 'ENCOUNTERID', 'AGE']].reset_index(), demo_onehot_cat], axis=1).drop('index',axis=1)    
    
    #Save table
    demo_one.to_pickle(datafolder+site+'/demo_'+site+'_'+str(year)+'.pkl')

    #consistency check
    if demo_one.empty:
        logging.basicConfig(filename='demo.log', filemode='a')    
        print('DATAFRAME EMPTY!!!!!! '+site+":"+str(year), flush = True)
        logging.error('demo: DATAFRAME EMPTY!!!!!! '+site+":"+str(year))
        logging.shutdown()

    print('Finished demo on site '+site+":"+str(year), flush = True)
    
def dx(configs_variables, year):
    '''
    The module process the diagnosis table for commorbidity (records before admission) 
    1) Include only all data before admission
    2) Translate ICD10 to ICD9 if possible
    3) Roll icd code to 3 digit
    4) Seperate Comorbidity into >6 months and <6 months before admission

    Input:
    p0_dx_{site}.pkl - Long format PCORNET table

    Output:
    dx_{site}_{str(year)}.pkl - One hot dx table (boolean)
    '''    

    # dx
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)    
    print('Running dx on site '+site+":"+str(year), flush = True)
    
    if not configs_variables['rerun_flag'] and os.path.exists(datafolder+site+'/dx_'+site+'_'+str(year)+'.pkl'):
        print('Existed: dx_'+site+'_'+str(year)+'.pkl')
        return        
    
    # load table
    dx = pd.read_parquet(datafolder+site+'/p0_dx_'+site+'_'+str(year)+'.parquet')
    dx['PATID'] = dx['PATID'].astype(str)
    dx['ENCOUNTERID'] = dx['ENCOUNTERID'].astype(str)    
    dx['DX_TYPE'] = dx['DX_TYPE'].astype(str)
    dx['DX_TYPE'] = dx['DX_TYPE'].str.replace('\.0','')
    dx = dx.drop_duplicates()
    
    #Some site use 9 some site use 09
    dx['DX_TYPE'] = dx['DX_TYPE'].where(dx['DX_TYPE'] != '9', '09')
    
    dx = dx[(dx['DX_TYPE']=='09') | (dx['DX_TYPE']=='10')]
    
    
    # Get the patient records in onset
    newdfX = pd.read_pickle(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl')
    newdfX = newdfX >> select('PATID', 'ENCOUNTERID', 'SINCE_ADMIT') >> mutate(dummy = True)
    dx = (pd.merge(dx, newdfX, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left').fillna({'dummy': False}) >> mask(X.dummy) >> select(~X.dummy)).reset_index(drop=True)

    #Only include historical records
    dx = dx[dx['DAYS_SINCE_ADMIT']<0].drop(['SINCE_ADMIT'],axis=1)
        

    
    # ICD10 -> ICD09
    icd10toicd09 = pd.read_csv(home_directory+'2018_I10gem.csv',sep=',')
    
    icd10toicd09.columns = ['DX', 'DX09']
    dx4 = dx >> mask(X.DX_TYPE == '10')
    dx4['DX'] = dx4['DX'].map(lambda x: x.replace('.',''))
    dx4 = dx4 >> left_join(icd10toicd09, by='DX')

    #Keep icd10 if no match
    dx4['DX_TYPE'] = dx4['DX_TYPE'].where(dx4['DX09'].isnull(), '09')
    dx4['DX'] = dx4['DX'].where(dx4['DX09'].isnull(), dx4['DX09'])
    dx4 = dx4.drop('DX09', axis=1)
    dx = pd.concat([dx >> mask(X.DX_TYPE != '10'), dx4], axis=0)

    # Roll icd 09 code up native
    dx['DX'] = dx['DX'].where(dx['DX_TYPE'] != '09', dx['DX'].map(lambda x: x[0:3]))

    # Roll icd 10 code up native
    dx['DX'] = dx['DX'].where(dx['DX_TYPE'] != '10', dx['DX'].map(lambda x: x[0:3]))
    
    # Transform dx table (Historical data: Yes if any diagnoasis show up)  (Boolean)
    # Fillna if no record
    dx['sixmonth'] = '<6'
    dx['sixmonth'] = dx['sixmonth'].where(dx['DAYS_SINCE_ADMIT']<-365/2, '>6') # becareful negative number
    
    dx_t = dx >> mutate(DX='DX:'+X.DX_TYPE+":"+X.DX+X.sixmonth) >> drop('DX_TYPE') >> drop('sixmonth')
#    dx_t = (dx_t >> drop('DAYS_SINCE_ADMIT') >> mutate(dummy = True) >> distinct()).pivot(index=['PATID', 'ENCOUNTERID'], columns='DX', values='dummy').fillna(False).reset_index()
    dx_t = (dx_t >> drop('DAYS_SINCE_ADMIT') >> mutate(dummy = True) >> distinct())[['PATID', 'ENCOUNTERID', 'DX', 'dummy']].drop_duplicates().pivot(index=['PATID', 'ENCOUNTERID'], columns='DX', values='dummy').fillna(False).reset_index()

    #Save table
    dx_t.to_pickle(datafolder+site+'/dx_'+site+'_'+str(year)+'.pkl')

    #consistency check
    if dx_t.empty:
        logging.basicConfig(filename='dx.log', filemode='a')
        print('DATAFRAME EMPTY!!!!!! '+site+":"+str(year), flush = True)
        logging.error('dx: DATAFRAME EMPTY!!!!!! '+site+":"+str(year))
        logging.shutdown()
    
    #consistency check2
        
    print('Finished dx on site '+site+":"+str(year), flush = True)
    
def px(configs_variables, year):
    '''
    The module process the procedure table to get the last avaliable data 24 hour before onset
    1) Include only all data before admission
    2) Drop all data before admit

    Input:
    p0_px_{site}.pkl - Long format PCORNET table

    Output:
    px_{site}_{str(year)}.pkl - One hot px table (boolean)
    '''    
    
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)
    print('Running px on site '+site+":"+str(year), flush = True)

    if not configs_variables['rerun_flag'] and os.path.exists(datafolder+site+'/px_'+site+'_'+str(year)+'.pkl'):
        print('Existed: px_'+site+'_'+str(year)+'.pkl')
        return            
    
    #load table
    px = pd.read_parquet(datafolder+site+'/p0_px_'+site+'_'+str(year)+'.parquet')
    px['PATID'] = px['PATID'].astype(str)
    px['ENCOUNTERID'] = px['ENCOUNTERID'].astype(str)        
    px['PX_TYPE'] = px['PX_TYPE'].astype(str)
    px = px.drop_duplicates()    
    
    # Get the patient records in onset
    # Calculate 'FUTURE' column as ONSET_DAY-MEASURE_DAY
    newdfX = pd.read_pickle(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl')
    newdfX = newdfX >> select('PATID', 'ENCOUNTERID', 'SINCE_ADMIT') >> mutate(dummy = True)
    px = (pd.merge(px, newdfX, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left').fillna({'dummy': False}) >> mask(X.dummy) >> select(~X.dummy) >> mutate(FUTURE=X.SINCE_ADMIT-X.DAYS_SINCE_ADMIT)).reset_index(drop=True)

    #24 hours prediction
    px = px[px['FUTURE']>0].drop(['SINCE_ADMIT','FUTURE'],axis=1)
    #Only include in-hosipital record
    px = px[px['DAYS_SINCE_ADMIT']>=0]
    
    #Some site use 9 some site use 09
    px['PX_TYPE'] = px['PX_TYPE'].where(px['PX_TYPE'] != '9', '09')
    
    # drop unused column
    px = px >> mutate(PX='PX:'+X.PX_TYPE+":"+X.PX) >> drop('PX_TYPE')

    # Transform px table (Boolean)
    # Fillna if no record
    px_t = (px >> drop('DAYS_SINCE_ADMIT') >> mutate(dummy = True) >> distinct())[['PATID', 'ENCOUNTERID', 'PX', 'dummy']].drop_duplicates().pivot(index=['PATID', 'ENCOUNTERID'], columns='PX', values='dummy').fillna(False).reset_index()

    #Save table
    px_t.to_pickle(datafolder+site+'/px_'+site+'_'+str(year)+'.pkl')

    if px_t.empty:
        logging.basicConfig(filename='px.log', filemode='a')
        print('DATAFRAME EMPTY!!!!!! '+site+":"+str(year), flush = True)
        logging.error('px: DATAFRAME EMPTY!!!!!! '+site+":"+str(year))
        logging.shutdown()


    print('Finished px on site '+site+":"+str(year), flush = True)
    
def lab(configs_variables, year):
    '''
    The module process the lab table to get the last avaliable data 24 hour before onset
    1) Include only all data 1 day before onset
    2) Drop all data before admit
    3) Calculate Daily Average
    4) Seperate into numeric lab and categoricl lab
    5) (Numeric) Calucalte Daily average
    6) Collect last avaliable data

    Input:
    p0_lab_g_{site}.pkl - Long format PCORNET table (unit unified)

    Output:
    lab_{site}_{str(year)}.pkl - One hot px table
    '''    

    site, datafolder, home_directory = utils_function.get_commons(configs_variables)    
    print('Running lab on site '+site+":"+str(year), flush = True)

    if not configs_variables['rerun_flag'] and os.path.exists(datafolder+site+'/labcat_'+site+'_'+str(year)+'.pkl') and os.path.exists(datafolder+site+'/labnum_'+site+'_'+str(year)+'.pkl'):
        print('Existed: labnum+labcat_'+site+'_'+str(year)+'.pkl')
        return     
    
    #load table
    if configs_variables['remove_outliner_flag']:    
        print('Loading remove outliner lab')        
        lab = pd.read_parquet(datafolder+site+'/p0_lab_g_'+site+'_'+str(year)+'_nooutliner.parquet')
    else:
        lab = pd.read_parquet(datafolder+site+'/p0_lab_g_'+site+'_'+str(year)+'.parquet')

    lab['PATID'] = lab['PATID'].astype(str)
    lab['ENCOUNTERID'] = lab['ENCOUNTERID'].astype(str)        
    lab = lab[lab['LAB_LOINC'].notnull()]
    
    nan_valueas = ['NI', 'UN', 'OT', 'None']
    replacenavalues = {x: None for x in nan_valueas}
    lab['RESULT_QUAL'] = lab['RESULT_QUAL'].replace(replacenavalues)
    
    # Get the patient records in onset
    # Calculate 'FUTURE' column as ONSET_DAY-MEASURE_DAY
    newdfX = pd.read_pickle(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl')
    newdfX = newdfX >> select('PATID', 'ENCOUNTERID', 'SINCE_ADMIT') >> mutate(dummy = True)
    lab = (pd.merge(lab, newdfX, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left').fillna({'dummy': False}) >> mask(X.dummy) >> select(~X.dummy) >> mutate(FUTURE=X.SINCE_ADMIT-X.DAYS_SINCE_ADMIT)).reset_index(drop=True)

    #24 hours prediciton
    lab = lab[lab['FUTURE']>0].drop(['SINCE_ADMIT','FUTURE'],axis=1)
    #Only include in-hosipital record
    lab = lab[lab['DAYS_SINCE_ADMIT']>=0]
    
    #seperate into numberic lab and pos/neg lab
    #calculate categorical first
    lab_cat = lab.loc[(lab['RESULT_NUM'].isnull()) & (lab['RESULT_QUAL'].notnull())]
    lab_cat = lab_cat.drop_duplicates()
    
    if lab_cat.empty:
        labcat_t = lab >> select('PATID','ENCOUNTERID')
        labcat_t.to_pickle(datafolder+site+'/labcat_'+site+'_'+str(year)+'.pkl')
    else:
        lab_mode = lab_cat.loc[:, ['PATID', 'ENCOUNTERID', 'LAB_LOINC', 'DAYS_SINCE_ADMIT', 'RESULT_QUAL']].groupby(['PATID', 'ENCOUNTERID', 'LAB_LOINC', 'DAYS_SINCE_ADMIT']).agg(pd.Series.mode).reset_index()
        lab_mode_nnd = lab_mode.loc[lab_mode['RESULT_QUAL'].apply(type) == str].copy()
        lab_mode_nd = lab_mode.loc[lab_mode['RESULT_QUAL'].apply(type) != str].copy()
        pattern = '[\[\]\']'
        lab_mode_nd.loc[:,'RESULT_QUAL'] = lab_mode_nd['RESULT_QUAL'].apply(lambda x: re.sub(pattern, "", np.array2string(x,separator='-')))
        lab_mode = pd.concat([lab_mode_nd, lab_mode_nnd], ignore_index=True)

        labcat_t = lab_mode.sort_values(['PATID', 'ENCOUNTERID', 'LAB_LOINC', 'DAYS_SINCE_ADMIT']).groupby(['PATID', 'ENCOUNTERID', 'LAB_LOINC']).agg({'RESULT_QUAL':'last'}).reset_index()
        labcat_t = labcat_t >> mutate(LAB_LOINC='LAB:'+":"+X.LAB_LOINC+"("+X.RESULT_QUAL+")") >> mutate(dummy = True) >> select('PATID', 'ENCOUNTERID', 'LAB_LOINC', 'dummy')
        labcat_t = labcat_t.pivot(index=['PATID', 'ENCOUNTERID'], columns='LAB_LOINC', values='dummy').fillna(False).reset_index()        
        #Save table    
        labcat_t.to_pickle(datafolder+site+'/labcat_'+site+'_'+str(year)+'.pkl') 
    
    #calculate numerica    
    lab_num = lab.loc[lab['RESULT_NUM'].notnull()]  
    if lab_num.empty:
        labnum_t = lab >> select('PATID','ENCOUNTERID')
        labnum_t.to_pickle(datafolder+site+'/labnum_'+site+'_'+str(year)+'.pkl')        
    else:    
        #Calculate daily average
        lab_mean = lab_num.groupby(['PATID', 'ENCOUNTERID', 'LAB_LOINC', 'RESULT_UNIT', 'DAYS_SINCE_ADMIT']).agg({'RESULT_NUM':'mean'}).reset_index()
        lab_mean = lab_mean >> mutate(LAB_LOINC='LAB:'+":"+X.LAB_LOINC+"("+X.RESULT_UNIT+")")
        labnum_t = lab_mean.sort_values(['PATID', 'ENCOUNTERID', 'LAB_LOINC', 'DAYS_SINCE_ADMIT']).groupby(['PATID', 'ENCOUNTERID', 'LAB_LOINC']).agg({'RESULT_NUM':'last'}).reset_index()
        labnum_t = labnum_t.pivot(index=['PATID', 'ENCOUNTERID'], columns='LAB_LOINC', values='RESULT_NUM').reset_index()
        #Save table
        labnum_t.to_pickle(datafolder+site+'/labnum_'+site+'_'+str(year)+'.pkl')


    if labnum_t.empty or labcat_t.empty:
        logging.basicConfig(filename='lab.log', filemode='a')
        print('DATAFRAME EMPTY!!!!!! '+site+":"+str(year), flush = True)
        logging.error('lab: DATAFRAME EMPTY!!!!!! '+site+":"+str(year))
        logging.shutdown()   

    print('Finished lab on site '+site+":"+str(year), flush = True)
    
def amed(configs_variables, year):
    '''
    The module process the amed table for medication to get the last avaliable data 24 hour before onset
    1) Include only all data 1 day before onset
    2) Drop all data before admit
    3) Convert rxnorm to atc code
    4) Convert ndc -> atc code

    Input:
    p0_amed_{site}.pkl - Long format PCORNET table (unit unified)

    Output:
    amed_{site}_{str(year)}.pkl - One hot px table
    '''    

    site, datafolder, home_directory = utils_function.get_commons(configs_variables)

    if not configs_variables['rerun_flag'] and os.path.exists(datafolder+site+'/amed_'+site+'_'+str(year)+'.pkl'):
        print('Existed: amed_'+site+'_'+str(year)+'.pkl')
        return     
    
    print('Running amed on site '+site+":"+str(year), flush = True)

    #load table
    amed = pd.read_parquet(datafolder+site+'/p0_amed_'+site+'_'+str(year)+'.parquet')
    amed['PATID'] = amed['PATID'].astype(str)
    amed['ENCOUNTERID'] = amed['ENCOUNTERID'].astype(str)        
    amed = amed.drop_duplicates()
    
    # Get the patient records in onset
    # Calculate 'FUTURE' column as ONSET_DAY-MEASURE_DAY
    newdfX = pd.read_pickle(datafolder+site+'/onset_'+site+'_'+str(year)+'.pkl')
    newdfX = newdfX >> select('PATID', 'ENCOUNTERID', 'SINCE_ADMIT') >> mutate(dummy = True)
    amed = (pd.merge(amed, newdfX, left_on=['PATID', 'ENCOUNTERID'], right_on=['PATID', 'ENCOUNTERID'], how='left').fillna({'dummy': False}) >> mask(X.dummy) >> select(~X.dummy) >> mutate(FUTURE=X.SINCE_ADMIT-X.DAYS_SINCE_ADMIT)).reset_index(drop=True)

    #24 hours prediction
    amed = amed[amed['FUTURE']>0].drop(['SINCE_ADMIT','FUTURE'],axis=1)
    #Only include in-hosipital record
    amed = amed[amed['DAYS_SINCE_ADMIT']>=0]
        
    # ndc -> rxnorm
    amed_rx = amed.loc[amed['MEDADMIN_TYPE'] == "RX"]
    amed_ndc = amed.loc[amed['MEDADMIN_TYPE'] == "ND"]    
    if not amed_ndc.empty:    
        ndc2rx = pd.read_parquet(datafolder+'/med_unified_conversion_nd2rx.parquet') >> rename(MEDADMIN_CODE=X.ND)
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
        rxcui2atc = pd.read_parquet(datafolder+'/med_unified_conversion_rx2atc.parquet') >> rename(MEDADMIN_CODE=X.RX)
        amed_rx = amed_rx >> left_join(rxcui2atc, by='MEDADMIN_CODE')
        amed_rx['MEDADMIN_TYPE'] = amed_rx['MEDADMIN_TYPE'].where(amed_rx['ATC'].isnull(), 'ATC')
        amed_rx['MEDADMIN_CODE'] = amed_rx['MEDADMIN_CODE'].where(amed_rx['ATC'].isnull(), amed_rx['ATC'])   

    # Recombine and reseperate
    amed = pd.concat([amed_rx, amed_ndc], axis=0, ignore_index=True)

    amed = amed >> mutate(MEDADMIN_CODE='MED:'+X.MEDADMIN_TYPE+':'+X.MEDADMIN_CODE)
    amed = amed >> select('PATID', 'ENCOUNTERID', 'MEDADMIN_CODE', 'DAYS_SINCE_ADMIT')

#     # rxnorm -> atc
#     amed_rx = amed.loc[amed['MEDADMIN_TYPE'] == "RX"]
#     if not amed_rx.empty:
#         # pd.DataFrame(amed['MEDADMIN_CODE'].unique()).to_csv('/home/hchan2/AKI/AKI_Python/data/'+site+'/rxnormtmp.csv', sep=',', index=False, header = False)
#         # Go to run rxnorm2atcR.ipynb NOW
#         rxcui2atc_dtypes =  {"Rxcui": 'object', "ATC4th": 'object'}    
#         rxcui2atc = pd.read_csv(datafolder+site+'/rxnorm_out_'+site+'.csv',sep=',', dtype=(rxcui2atc_dtypes)) >> rename(MEDADMIN_CODE=X.Rxcui)
#         amed_rx = amed_rx >> left_join(rxcui2atc, by='MEDADMIN_CODE')
#         amed_rx['MEDADMIN_TYPE'] = amed_rx['MEDADMIN_TYPE'].where(amed_rx['ATC4th'].isnull(), 'ATC')
#         amed_rx['MEDADMIN_CODE'] = amed_rx['MEDADMIN_CODE'].where(amed_rx['ATC4th'].isnull(), amed_rx['ATC4th'])
#         amed_rx = amed_rx >> mutate(MEDADMIN_CODE='MED:'+X.MEDADMIN_TYPE+':'+X.MEDADMIN_CODE)
#         amed_rx = amed_rx >> select('PATID', 'ENCOUNTERID', 'MEDADMIN_CODE', 'DAYS_SINCE_ADMIT')
#     else:
#         amed_rx = amed_rx >> select('PATID', 'ENCOUNTERID', 'MEDADMIN_CODE', 'DAYS_SINCE_ADMIT')
    
#     # ndc -> atc
#     amed_ndc = amed.loc[amed['MEDADMIN_TYPE'] == "ND"]    
#     if not amed_ndc.empty:    
#         # pd.DataFrame(amed['MEDADMIN_CODE'].unique()).to_csv('/home/hchan2/AKI/AKI_Python/data/'+site+'/rxnormtmp.csv', sep=',', index=False, header = False)
#         # Go to run rxnorm2atcR.ipynb NOW
#         ndc2atc_dtypes =  {"ndc": 'object', "ATC4th": 'object'}    
#         ndc2atc = pd.read_csv(datafolder+site+'/ndc_out_'+site+'.csv',sep=',', dtype=(ndc2atc_dtypes)) >> rename(MEDADMIN_CODE=X.ndc)
#         amed_ndc = amed_ndc >> left_join(ndc2atc, by='MEDADMIN_CODE')
#         amed_ndc['MEDADMIN_TYPE'] = amed_ndc['MEDADMIN_TYPE'].where(amed_ndc['ATC4th'].isnull(), 'ATC')
#         amed_ndc['MEDADMIN_CODE'] = amed_ndc['MEDADMIN_CODE'].where(amed_ndc['ATC4th'].isnull(), amed_ndc['ATC4th'])
#         amed_ndc = amed_ndc >> mutate(MEDADMIN_CODE='MED:'+X.MEDADMIN_TYPE+':'+X.MEDADMIN_CODE)
#         amed_ndc = amed_ndc >> select('PATID', 'ENCOUNTERID', 'MEDADMIN_CODE', 'DAYS_SINCE_ADMIT')
#     else:
#         amed_ndc = amed_ndc >> select('PATID', 'ENCOUNTERID', 'MEDADMIN_CODE', 'DAYS_SINCE_ADMIT')
       
#     amed = pd.concat([amed_rx, amed_ndc], axis=0, ignore_index=True)   

    
    # Transform amed table (Boolean)
    # Fillna if no record    
    amed_t = (amed >> drop('DAYS_SINCE_ADMIT') >> mutate(dummy = True) >> distinct()).pivot(index=['PATID', 'ENCOUNTERID'], columns='MEDADMIN_CODE', values='dummy').fillna(False).reset_index()
        
    #Save table
    amed_t.to_pickle(datafolder+site+'/amed_'+site+'_'+str(year)+'.pkl')

    if amed_t.empty:
        logging.basicConfig(filename='amed.log', filemode='a')
        print('DATAFRAME EMPTY!!!!!! '+site+":"+str(year), flush = True)
        logging.error('amed: DATAFRAME EMPTY!!!!!! '+site+":"+str(year))
        logging.shutdown()   

    print('Finished amed on site '+site+":"+str(year), flush = True)
    
def unify_lab(configs_variables):
    '''
    The module process convert lab units and attempt to group LONIC into groups into common units if possible

    Input:
    p0_lab_{site}.pkl - Long format PCORNET table (unit unified)

    Output:
    p0_lab_g_{site}}.pkl - One hot px table
    '''        
    site, datafolder, home_directory = utils_function.get_commons(configs_variables)    
    print('Running unify lab on site '+site, flush = True)

    if not configs_variables['rerun_flag'] and os.path.exists(datafolder+site+'/p0_lab_g_'+site+'.parquet'):
        print('Existed: p0_lab_g_'+site+'.parquet')
        return     
    
    UCUMunitX = pd.read_csv('/home/hoyinchan/code/AKI_CDM_PY/UCUMunitX.csv')
    UCUMunitX = UCUMunitX[UCUMunitX['factor_final'].notna()]
    UCUMunitX = UCUMunitX.drop('Unnamed: 0',axis=1).drop_duplicates()
    
    
    local_custom_convert = pd.read_csv('/home/hoyinchan/code/AKI_CDM_PY/local_custom_convert.csv')
    local_custom_convert = local_custom_convert.drop('Unnamed: 0',axis=1).drop_duplicates()
    
    UCUMqualX = pd.read_csv('/home/hoyinchan/code/AKI_CDM_PY/UCUMqualX.csv')
    UCUMqualX = UCUMqualX.drop('Unnamed: 0',axis=1).drop_duplicates()
    
    
    loincmap3 =pd.read_csv(home_directory+'loinc/AccessoryFiles/GroupFile/GroupLoincTerms.csv') 
    
#    labtest = pd.read_parquet(datafolder+site+'/p0_lab_'+site+'.parquet')
    labtest = pd.read_parquet(datafolder+site+'/p0_lab_'+site+'.parquet')
    
    labtest['site']=site

    labtest2 = labtest.merge(local_custom_convert, left_on = ['LAB_LOINC', 'site', 'RESULT_UNIT'], right_on = ['LAB_LOINC', 'site', 'SOURCE_UNIT'], how='left')
    labtest2['NEW_UNIT'] = np.where(labtest2['TARGET_UNIT'].notnull(), labtest2['TARGET_UNIT'], labtest2['RESULT_UNIT'])
    labtest2['NEW_RESULT_NUM'] = np.where(labtest2['TARGET_UNIT'].notnull(), labtest2['Multipliyer']*labtest2['RESULT_NUM'], labtest2['RESULT_NUM'])

    labtest3 = labtest2.copy()
    labtest3['RESULT_UNIT'] = labtest3['NEW_UNIT']
    labtest3['RESULT_NUM'] = labtest3['NEW_RESULT_NUM']
    labtest3 = labtest3.drop(['NEW_UNIT', 'NEW_RESULT_NUM', 'SOURCE_UNIT', 'TARGET_UNIT', 'LONG_COMMON_NAME', 'Multipliyer'], axis=1)

    labtest4 = labtest3.merge(UCUMunitX, on = ['LAB_LOINC', 'RESULT_UNIT'], how='left').copy()
    labtest4['NEW_UNIT'] = np.where(labtest4['FINAL_UNIT'].notnull(), labtest4['FINAL_UNIT'], labtest4['RESULT_UNIT'])
    labtest4['NEW_RESULT_NUM'] = np.where(labtest4['FINAL_UNIT'].notnull(), labtest4['factor_final']*labtest4['RESULT_NUM'], labtest4['RESULT_NUM'])
    labtest4['NEW_LAB_LOINC'] = np.where(labtest4['FINAL_UNIT'].notnull(), labtest4['GroupId'], labtest4['LAB_LOINC'])
    labtest4['RESULT_UNIT'] = labtest4['NEW_UNIT']
    labtest4['RESULT_NUM'] = labtest4['NEW_RESULT_NUM']
    labtest4['LAB_LOINC'] = labtest4['NEW_LAB_LOINC']
    labtest4 = labtest4.drop(['GroupId', 'EXAMPLE_UCUM_UNITS',
           'EXAMPLE_UCUM_UNITS_FINAL', 'RESULT_UNIT_CONSENSUS', 'FINAL_UNIT',
           'FINAL_Multiplyer', 'RESULT_UNIT_API', 'FINAL_UNIT_API', 'factor_final',
           'NEW_UNIT', 'NEW_RESULT_NUM', 'NEW_LAB_LOINC'], axis=1)

#    mmc = loincmap3[loincmap3['Category']=='Mass-Molar conversion'][['GroupId']]
#    labtest4 = labtest4.merge(mmc, left_on = 'LAB_LOINC', right_on='GroupId', how='left', indicator=True)
#    labtest4 = labtest4[labtest4['_merge']=='left_only']
#    labtest4 = labtest4.drop(['GroupId', '_merge'],axis=1)

    labtest5 = labtest4.copy()
    labtest5 = labtest5.merge(UCUMqualX[['LAB_LOINC', 'GroupId']].drop_duplicates(), on='LAB_LOINC', how='left')
    labtest5['NEW_LAB_LOINC'] = np.where(labtest5['GroupId'].notnull(), labtest5['GroupId'], labtest5['LAB_LOINC'])
    labtest5['LAB_LOINC'] = labtest5['NEW_LAB_LOINC']
    labtest5 = labtest5.drop(['GroupId','NEW_LAB_LOINC'],axis=1)
    labtest5 = labtest5.drop('site',axis=1)
    labtest5 = labtest5.drop_duplicates()
    labtest5.to_parquet(datafolder+site+'/p0_lab_g_'+site+'.parquet')
    
    print('Finished unify lab on site '+site, flush = True)
    
