#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:46:46 2023

Open downloaded sst files,
coarsen them to 10 degrees
calculate teleconnections to each individual basin
you need to merge the dataframes somehow. basically the basin should jus tbe treated as something 
do one sst patch per time
check you are doing the climatology and anomaly calculation right
simplify s2s forecaster object since s2s_forecaster might have functions you will not use, or mayube make a simpler object
10-06-23: if u dont go w T do some household chores like clothes and tomorrow work in the morning and workout
@author: amirouyed

https://data.giss.nasa.gov/gistemp/


"""

import xarray as xr
from s2s_forecaster import s2s_forecaster
from tqdm import tqdm
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
    
    
def old_corr_algorithm(ds_prism, ds_sst, var_label):   
    for lead_month in tqdm([2]):
        for season in tqdm(['DJF','MAM','JJA','SON']):
            d = {'basin':[],'lat': [], 'lon': [], 'corr': []}

        #for season in tqdm(['DJF']):
            for basin in tqdm(ds_prism['basin'].values):
                for lat in ds_sst['lat'].values:
                    for lon in ds_sst['lon'].values:
                        ds_unit= align_ds(ds_prism,ds_sst,lat, lon, lead_month, season,basin, var_label)
                
                        #print(ds_prism)
                        corr=  xr.corr(ds_unit[var_label+'_anom_basin'], ds_unit['tempanomaly']).item()
            
                        df = ds_unit[['tempanomaly', var_label, var_label+'_anom_basin']].to_dataframe().reset_index()
                        corr = df[[var_label+'_anom_basin', 'tempanomaly']].corr().values.ravel()[1]
                        #print(corr)
                        #d['season'].append(season)
                       # d['lead_month'].append(lead_month)
                        d['basin'].append(basin)
                        d['lat'].append(lat)
                        d['lon'].append(lon)
                        d['corr'].append(corr)
        
                    
            df=pd.DataFrame(data=d)
            df=df.set_index(['basin','lat','lon'])
            print(df)
            ds_corr=xr.Dataset.from_dataframe(df)
            ds_corr.to_netcdf('../data/processed/ds_corr_'+ season+'_lead_'+str(lead_month)+'.nc')
            #ds_corr=ds_corr.sel(basin=2,season='DJF',lead_month=0)
            ds_corr['corr'].plot()
            plt.show()
            plt.close()


def normalized_ds(ds_prism_0, var_label, mid_month_label=False):
    deltat = np.timedelta64(15, 'D')
    ds_prism = ds_prism_0.resample(time='M', label='left').mean()
    if mid_month_label:
        dates=ds_prism['time'].values
        new_dates = dates.astype('datetime64[M]') + np.timedelta64(14, 'D')
        ds_prism = ds_prism.assign_coords(time=new_dates)
    climatology = ds_prism.groupby('time.month').mean(dim='time')
    ds_prism['T_anom'] = ds_prism['T'].groupby('time.month') - climatology['T']
    return ds_prism
    
def align_ds(ds,ds_sst, lat, lon, lead_month, season,basin, var_label):
    ds_basin = ds.sel(time=ds.time.dt.season == season, basin=basin)
    deltat = np.timedelta64(lead_month*30, 'D')
    times=ds_basin['time'].values - deltat
    ds_unit = ds_sst.sel(lat=lat, lon=lon, time=times, method='nearest')
    ds_basin = ds_basin.assign_coords(time=ds_unit['time'].values)

    ds_unit[var_label+'_anom_basin'] = ds_basin[var_label+'_anom']
    ds_unit[var_label] = ds_basin[var_label]
    return ds_unit



def normalize_ds(ds):
    climatology_mean = ds.groupby("time.month").mean("time")
    anomalies = ds.groupby("time.month") - climatology_mean
    for var in ds: 
        ds[var+'_anom']=ds[var].groupby("time.month")-climatology_mean[var]
        ds[var+'_climatology']=climatology_mean[var]

    return ds

def main():
    ds_sst = xr.open_dataset('../data/raw/gistemp1200_GHCNv4_ERSSTv5.nc')
    ds_sst = ds_sst.coarsen(lat=5, lon=5, boundary='trim').mean()
    
    season = 'SON'
    
    ds_swe=xr.open_dataset('../data/raw/PRISM_UA_data_broxon.nc')
    dates = pd.to_datetime( pd.DataFrame({'year': ds_swe['Year'].values, 'month': ds_swe['Month'].values, 'day':15}))
    ds_swe=ds_swe.assign_coords(time=dates.values)
    ds_swe=ds_swe.rename({'watershed':'basin'})
    #ds_swe['datetime']=(['time'], dates.values)
    ds_swe=normalize_ds(ds_swe)
    
    #s2s = s2s_forecaster(lead=0, roll_dt=3, seas=season)
    # ds_prism_0 = s2s.ds_prism
    # ds_prism_0 = ds_prism_0.drop('month')
    # ds_prism_0 = ds_prism_0.drop('T_anom')
    # ds_prism=normalized_ds(ds_prism_0, False)
    # ds_prism_mid_month_labels=normalized_ds(ds_prism_0, True)

    ds_unit=align_ds(ds_swe, ds_sst,0,0,1,'DJF',1, 'SWE')
    #ds_unit2=align_ds(ds_prism_mid_month_labels, ds_sst,0,0,1,'DJF',1)

    
    old_corr_algorithm(ds_swe, ds_sst, 'SWE')
    
if __name__ == "__main__":
    main()

    
    # ds_prism=ds_prism.rename({'T':'T_anom'})
    
    
