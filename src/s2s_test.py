#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:31:57 2022

@author: aouyed
"""
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from s2s_forecaster import s2s_forecaster


def plot_series(df_pred,df_prism, df_gefs,  var):
    fig, ax = plt.subplots()

    
    ax.plot(df_gefs['time'], df_gefs[var], '-o', label='gefs')
    ax.plot(df_pred['time'], df_pred[var], '-o', label='pred')
    ax.plot(df_prism['time'], df_prism[var], '-o', label='prism')


    ax.legend(frameon=None)
    ax.set_xlabel("time")
    ax.set_ylabel(var)
    ax.set_title(var)
    directory = '../data/processed/plots/ts_'+var
    plt.savefig(directory+'.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    

s2s=s2s_forecaster(lead=1)
ds_pred=s2s.ds_pred_linear
ds_prism=s2s.ds_prism
ds_gefs=s2s.ds_gefs


ds_pred=ds_pred.sel(basin=18)
ds_prism=ds_prism.sel(basin=18)
ds_gefs=ds_gefs.sel(basin=18)

ds_prism=ds_prism.sel(time=ds_pred['time'].values)
ds_gefs=ds_gefs.sel(time=ds_pred['time'].values)


df_pred=ds_pred.to_dataframe().reset_index().drop('basin', axis='columns')
df_prism=ds_prism.to_dataframe().reset_index().drop('basin',axis='columns')
df_gefs=ds_gefs.to_dataframe().reset_index().drop('basin',axis='columns')
df_pred_error=abs(df_pred-df_prism)
df_gefs_error=abs(df_gefs-df_prism)
df_pred_error['time']=df_pred['time'].values
df_gefs_error['time']=df_gefs['time'].values


plot_series(df_pred,df_prism, df_gefs, 'T_anom')
plot_series(df_pred,df_prism, df_gefs, 'P_anom')
plot_series(df_pred,df_prism, df_gefs, 'SWE_anom')
plot_series(df_pred,df_prism, df_gefs, 'T')
plot_series(df_pred,df_prism, df_gefs, 'P')
plot_series(df_pred,df_prism, df_gefs, 'SWE')

plot_series(df_pred_error,df_prism, df_gefs_error, 'T_anom')
plot_series(df_pred_error,df_prism, df_gefs_error, 'P_anom')
plot_series(df_pred_error,df_prism, df_gefs_error, 'SWE_anom')