#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:10:19 2022

@author: aouyed
"""

import xarray as xr
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
from s2s_forecaster import s2s_forecaster
from pandas.plotting import autocorrelation_plot
import dask
import datetime

def big_histogram(x,y, xedges, yedges, bins=100):
     
    xbins = np.linspace(xedges[0], xedges[1], bins+1)
    ybins = np.linspace(yedges[0], yedges[1], bins+1)
    heatmap = np.zeros((bins, bins), np.uint)
    
        
    heatmap, _, _ = np.histogram2d(
                x, y, bins=[xbins, ybins])
    heatmap = 100*heatmap/np.sum(heatmap)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent




def hist_unit(dfx,dfy, labelx, labely, xedges= [-10,10], yedges= [-10,10],bins=10):
    
    fig, ax= plt.subplots()
    hist,edges=big_histogram(dfx[labelx].values,dfy[labely].values, xedges, yedges, bins)
    im = ax.imshow(hist, vmin=0, vmax=1, extent=edges,aspect='auto',origin='lower', cmap='CMRmap_r')
    plt.tight_layout()
    #fig.subplots_adjust(hspace=0.15)

    plt.savefig('../data/processed/plots/2d_hist_'+ labelx+'_'+labely+'.png', bbox_inches='tight', dpi=500)
    plt.show()
    plt.close()



def his2d(dfx,dfy,labelx,labely):
    xedges=[dfx[labelx].quantile(0.01),dfx[labelx].quantile(0.99)]
    yedges=[dfy[labely].quantile(0.01),dfy[labely].quantile(0.99)]
    
    hist_unit(dfx,dfy,labelx,labely, xedges= xedges, yedges= yedges,bins=25)
        



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


def plot_series_1d(df,var):
    fig, ax = plt.subplots()

    
    ax.plot(df['time'], df[var], '-o', label=var)
   
    ax.legend(frameon=None)
    ax.set_xlabel("time")
    ax.set_xlim([datetime.date(2014, 1, 1), datetime.date(2016, 1, 1)])

    ax.set_ylabel(var)
    ax.set_title(var)
    directory = '../data/processed/plots/ts_'+var
    plt.savefig(directory+'.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()    


    

def main():  
    s2s=s2s_forecaster(lead=1)
    ds_pred=s2s.ds_pred_linear
    ds_prism=s2s.ds_prism
    ds_gefs=s2s.ds_gefs
    ds_prism=s2s.teleconnect_total(ds_prism)
    for tel in ['NINO1+2', 'ANOM12', 'NINO3', 'ANOM3', 'NINO4', 'ANOM4', 'NINO3.4', 'ANOM3.4','anom']:
        ds_corr=s2s.ds_correlate(ds_prism, var2=tel, season='MAM')
        print(tel)
        print(ds_corr_djf.max())
    breakpoint()
    ds_pred=ds_pred.sel(basin=18)
    ds_prism=ds_prism.sel(basin=18)
    ds_gefs=ds_gefs.sel(basin=18)
    dfx=ds_prism.to_dataframe()
    dfy=ds_gefs.to_dataframe()
    

    dfx.loc[:,'month']=pd.DatetimeIndex(dfx.reset_index()['time']).month
    dfx.loc[:,'year']=pd.DatetimeIndex(dfx.reset_index()['time']).year
    #dfy=dfx.copy()
    dfx=dfy.copy()
    #dfx=s2s.teleconnect2(dfx)

    #dfy=s2s.teleconnect2(dfy)

    #dfy=ds_gefs.to_dataframe()
    #dfy=dfy-dfx
    
    labelx='week'
    labely='T_std'
    dfy=dfy.reset_index()
    

    
    #plot_series_1d(dfx,labely)
    # plot_series_1d(dfx,'P_anom')
    # plot_series_1d(dfx,'T_anom')
    
    # plot_series_1d(dfy,'P_std')
    # plot_series_1d(dfy,'T_std')
    # plot_series_1d(dfy,'P_skew')
    # plot_series_1d(dfy,'T_skew')
    # plot_series_1d(dfy,'P_kurtosis')
    # plot_series_1d(dfy,'T_kurtosis')


    his2d(dfx,dfy,labelx,labely)
    #autocorrelation_plot(dfy['SWE_anom'])
    
    
if __name__ == '__main__':
    main()
    