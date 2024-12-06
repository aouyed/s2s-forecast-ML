#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 17:06:05 2021

@author: aouyed
"""


import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bias_correction import BiasCorrection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor

pd.options.mode.chained_assignment = None  # default='warn'

def preprocess_ds(lead):
    ds=xr.open_dataset('../data/raw/GEFS_and_Observations_of_P_T_SWE_SM.nc')
    ds_prism=ds[['PRISM_P','PRISM_T','UA_SWE']]
    ds_prism=ds_prism.rename({'PRISM_P':'P','PRISM_T': 'T','UA_SWE':'SWE'})
    ds_gefs=ds[['GEFS_P','GEFS_T','GEFS_SWE']]
    ds_gefs=ds_gefs.rename({'GEFS_P':'P','GEFS_T': 'T','GEFS_SWE':'SWE'})
    ds_gefs=ds_gefs.loc[{'lead':lead}].copy()
    ds_gefs['P']=ds_gefs['P'].mean(dim='ensemble')
    ds_gefs['T']=ds_gefs['T'].mean(dim='ensemble')
    ds_gefs['SWE']=ds_gefs['SWE'].mean(dim='ensemble')
    ds_gefs=ds_gefs.drop('ensemble')
    return ds_prism, ds_gefs

def get_models():
    models = dict()
    models['linear'] = LinearRegression()
    models['rf'] = RandomForestRegressor()
    return models

def ds_split(ds_era_az, ds_gefs):
    times_train=ds_era_az['time'].values[:654]
    times_test=ds_era_az['time'].values[655:]
    
    ds_era_az_train=ds_era_az.sel(time=times_train)
    ds_era_gefs_train=ds_gefs.sel(time=times_train)

    ds_era_az_test=ds_era_az.sel(time=times_test)
    ds_era_gefs_test=ds_gefs.sel(time=times_test)

   
    
    #df_x.loc[:,'month']=pd.DatetimeIndex(df_x.reset_index()['time']).month
    return ds_era_az_train, ds_era_gefs_train, ds_era_az_test,  ds_era_gefs_test


def convert_ds_to_df(ds):
    df_x.loc[:,'month']=pd.DatetimeIndex(df_x.reset_index()['time']).month
    return df_x

def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def teleconnect(df_x):
    df_i=pd.read_csv('../data/raw/ersst5.nino.mth.81-10.ascii', sep='   ', engine='python')
    df_i=df_i.drop(['NINO3.4','NINO1+2','NINO3','NINO4'],axis=1)
    df_i['month']=df_i['month']+1
    df_i.loc[df_i.month==13,'month']=1
    df_x= pd.merge(df_x,df_i,how='left',on=['month','year'])
    #df_i=pd.read_csv('../data/raw/norm.nao.monthly.b5001.current.ascii.txt', sep='   ',engine='python')
    #df_i['month']=df_i['month']+1
    #df_i.loc[df_i.month==13,'month']=1
    #df_x= pd.merge(df_x,df_i,how='left',on=['month','year'])
    df_x=df_x.drop('year',axis=1)
    return df_x


def df_processor(ds_era_az, ds_gefs, basin):
    ds_x=ds_gefs.sel(basin=basin)
    ds_y=ds_era_az.sel(basin=basin)
    clim=ds_y[['T','P','SWE','T_climatology','P_climatology','SWE_climatology','time']].to_dataframe().reset_index()

    df_x=ds_x[['T_anom','P_anom','SWE_anom']].to_dataframe().reset_index()
    df_y=ds_y[['T_anom','P_anom','SWE_anom']].to_dataframe().reset_index()        
    df_x.loc[:,'month']=pd.DatetimeIndex(df_x.reset_index()['time']).month
    df_x.loc[:,'year']=pd.DatetimeIndex(df_x.reset_index()['time']).year

    df_x=df_x[['T_anom','P_anom','SWE_anom','month','week','year']]
    df_y=df_y[['T_anom','P_anom','SWE_anom']]
    
    return df_x, df_y, clim
    
def ml_calculator(self, ds_gefs, regressor, tele):
    
    ds_era_az_train, ds_gefs_train, ds_era_az_test,  ds_gefs_test=ds_split(ds_era_az, ds_gefs)
    y_pred_total=pd.DataFrame()
    regressor=LinearRegression()
    for basin in ds_era_az['basin'].values:
        print(basin)
        x_test, y_test, clim_test=df_processor(ds_era_az_test, ds_gefs_test, basin)
        x_train, y_train,clim_train=df_processor(ds_era_az_train, ds_gefs_train, basin)
        if tele:
            x_test=teleconnect(x_test)
            x_train=teleconnect(x_train)
        else:
            x_test=x_test.drop('year',axis=1)
            x_train=x_train.drop('year',axis=1)


        regressor.fit(x_train, y_train)
      
        y_pred = regressor.predict(x_test)
        #y_train = regressor.predict(x_train)
        y_pred=pd.DataFrame(y_pred, columns=['T_anom','P_anom','SWE_anom'])
        #y_train=pd.DataFrame(y_train, columns=['T_anom','P_anom','SWE_anom'])
        y_pred=pd.concat([y_pred,clim_test], axis=1)
        y_pred['basin']=basin
        if y_pred_total.empty:
            y_pred_total=y_pred
        else:
            y_pred_total=pd.concat([y_pred_total,y_pred])

    y_pred_total=y_pred_total.set_index(['basin','time'])        
    ds_pred=xr.Dataset.from_dataframe(y_pred_total)
    ds_error=(ds_pred - ds_era_az_test)**2
    return y_pred, ds_error, ds_era_az_test

def climatology(ds_era_az, ds_gefs, lead):
    climatology_mean = ds_era_az.groupby("time.week").mean("time")
    anomalies = ds_era_az.groupby("time.week") - climatology_mean
    ds_era_az_base=ds_era_az.copy()
    ds_gefs_base=ds_gefs.copy()
    for var in ds_era_az_base: 
        ds_era_az_base[var+'_anom']=ds_era_az[var].groupby("time.week")-climatology_mean[var]
        ds_era_az_base[var+'_climatology']=-ds_era_az_base[var+'_anom']+ds_era_az_base[var]
        ds_gefs_base[var+'_anom']=ds_gefs_base[var].groupby("time.week")-climatology_mean[var]
  
    return ds_era_az_base, ds_gefs_base


def bias_stage(rmses):
    print('one_stage')
    lead=LEAD
    regressor=RandomForestRegressor()
    print('lead week: ' + str(lead))
    ds_era_az, ds_gefs= pr.preprocess_ds(lead)
    df_x, df_y, rmses=pr.climatology(ds_era_az, ds_gefs,lead,  rmses)
    df_x= pr.preprocess_df(df_x)
    df_x=pr.teleconnect(df_x)
    X_train, X_test, y_train, y_test =train_test_split( df_x, df_y, test_size=TEST_SIZE,random_state=1)
    bias_correction(y_train, X_train[['P_anom','T_anom']], y_test, X_test[['P_anom','T_anom']])
    print('stats')



def bias_correction(y_train,X_train, y_test, X_test):
    label='P_anom'
    y_train=xr.Dataset.from_dataframe(y_train.reset_index(drop=True))
    X_train=xr.Dataset.from_dataframe(X_train.reset_index(drop=True))
    y_test=xr.Dataset.from_dataframe(y_test.reset_index(drop=True))
    X_test=xr.Dataset.from_dataframe(X_test.reset_index(drop=True))
    bc = BiasCorrection(y_train[label], X_train[label], X_test[label])
    df2 = bc.correct(method='modified_quantile')
    print('bias rmse')
    print('rmse: ' + str(np.sqrt(mean_squared_error(df2.to_dataframe().reset_index(drop=True)
    ,y_test[label].to_dataframe().reset_index(drop=True) ))))
    breakpoint()
    