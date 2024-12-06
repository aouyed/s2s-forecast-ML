#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:15:41 2022

@author: aouyed
"""

import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np 
import constants as c
from  s2s_forecaster import s2s_forecaster
import datetime


def plot_series_1d(df_x,df_y, df_z, vars_x, vars_y,tag_x,tag_y,tag_z, tag):
    fig, ax = plt.subplots()

    
    for var in vars_x:
        ax.plot(df_x['time'], df_x[var], '-o', label=var+tag_x)
    for var in vars_y:
        ax.plot(df_y['time'], df_y[var], '-o', label=var+tag_y)
        ax.plot(df_z['time'], df_z[var], '-o', label=var+tag_z)


   
    ax.legend(frameon=None)
    ax.set_xlabel("time")
    ax.set_xlim([datetime.date(2014, 1, 1), datetime.date(2016, 1, 1)])

    ax.set_ylabel(var)
    ax.set_title(var)
    directory = '../data/processed/plots/ts_'+var+tag
    plt.savefig(directory+'.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()   
    
    
def main():  
    basin=3
    season='SON'

    #s2s=s2s_forecaster(lead=2, varx=['P_anom','T_anom'],vary=['P_anom','T_anom'],
     #                  roll_dt=4, seas='SON')
     
     
     
    s2s=s2s_forecaster(lead=2, varx=['P_anom','T_anom'],vary=['P_anom','T_anom'],
                        roll_dt=4, seas='all')
    
   
    ds_prism=s2s.ds_prism
    ds_gefs=s2s.ds_gefs
    ds_pred_linear=s2s.ds_pred_linear
    ds_pred_stacked=s2s.ds_pred_stacked
    ds_error_linear=s2s.ds_error_linear
    ds_error_gefs=s2s.ds_error_gefs
    ds_error_stacked=s2s.ds_error_stacked

    df1=ds_prism.to_dataframe().reset_index()
    
    df2=ds_pred_stacked.to_dataframe().reset_index()
    df3=ds_gefs.to_dataframe().reset_index()
    varis=c.VARS
    tels=c.TELS
    all_vars=varis+tels
    vars1=['P_anom','NINO3.4']
    vars2=['P_anom']
    plot_series_1d(df1,df2,df3, vars1,vars2, '_prism','_stacked','_gefs','_')
    
    df1=ds_error_gefs.to_dataframe().reset_index()
    df2=ds_error_linear.to_dataframe().reset_index()
    df3=ds_error_stacked.to_dataframe().reset_index()
   
    varis=c.VARS
    tels=c.TELS
    all_vars=varis+tels
    vars1=['P_anom']
    vars2=['P_anom']
    plot_series_1d(df1,df2,df3, vars1,vars2, '_gefs','_linear','_stacked','_error')
    
    
    
if __name__ == '__main__':
    main()