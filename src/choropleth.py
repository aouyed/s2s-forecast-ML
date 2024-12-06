#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:54:39 2022

@author: aouyed
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import numpy as np
from  s2s_forecaster import s2s_forecaster
import xarray as xr
from sklearn.impute import SimpleImputer

import processors as pr


TELS=['NINO1+2', 'ANOM12', 'NINO3', 'ANOM3', 'NINO4', 'ANOM4', 'NINO3.4', 'ANOM3.4','NAO']
SEASONS=['DJF','MAM','JJA','SON']
SEASONS_tot=['DJF','MAM','JJA','SON','total']

def clim_ratio(df,varis):
    for var in varis:
        df[var+'_ratio']=df[var+'_anom']/df[var+'_clim']
  

def ds_correlate(ds,var1='P_anom',var2='NINO3.4', season='DJF'):
    imputer = SimpleImputer()
    df=ds.to_dataframe().reset_index()

    df[['P_anom','T_anom']]=imputer.fit_transform(df[['P_anom','T_anom']])
    df=df.set_index(['basin','time'])
    ds=xr.Dataset.from_dataframe(df)
    if season in SEASONS:

        ds= ds.sel(time=ds.time.dt.season==season)
        

    return xr.corr(ds[var1], ds[var2], dim="time")    

def choropleth(df, lead_str,var, vmin=-1, vmax=1, cmap_l='RdBu'):
    indexes=df['basin'].values
    index_str=[f"{i:02}" for i in indexes]
    df['basin_str']=np.array(index_str)
    fig=plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    ax.set_extent([-130, -60, 20, 55], ccrs.PlateCarree())
    ax.set_title('precipitation anomaly')
    cmap = plt.cm.get_cmap(cmap_l)
    norm = Normalize(
        vmin=vmin, vmax=vmax)
    #breakpoint()
    
    for s,t in df[['basin_str',var]].values:
        
        file='../data/raw/WBD/'+s+'/WBDHU2_US_s.shp'
        shape_feature = ShapelyFeature(Reader(file).geometries(),
                                    ccrs.PlateCarree(), edgecolor='black')
        ax.add_feature(shape_feature, facecolor=cmap(norm(t)))
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin,vmax))
    sm._A = []
    cbar_ax = fig.add_axes([0.1, 0.1, 0.78, 0.05])
    fig.colorbar(sm, ax=ax, cax=cbar_ax,orientation="horizontal", pad=0.3, label='Δ RMSE [mm]')
    #fig.colorbar(ax, cax=cbar_ax,orientation="horizontal", pad=0.5, label=units_label) 
    fig.tight_layout(rect=[0, 1.3, 1, 1.2])
    #fig.tight_layout()
    #fig.subplots_adjust(bottom=0.3)
    plt.savefig('../data/processed/plots/'+var +'_l'+lead_str+'.png', dpi=300)
    plt.show()
    plt.close()
        

def main():  
    lead=2
    season='SON'
    #var='P_anom'
    
    s2s=s2s_forecaster(lead=lead, roll_dt=3, seas=season)
    ds_prism=s2s.ds_prism
    ds_rmse_stacked=s2s.ds_rmse_stacked
    ds_rmse_gefs=s2s.ds_rmse_gefs

    ds_rmse_linear=s2s.ds_rmse_linear
    ds_rmse_rf=s2s.ds_rmse_rf

    ds_delta_rmse=ds_rmse_rf-ds_rmse_gefs
    for var in ['T_anom','P_anom']:
        for season in [season]:
            #for tel in ['NAO','NINO3.4','ANOM3.4']:
                #ds_corr=ds_correlate(ds_prism,var1=var, var2=tel, season=season)
                
                    #ds_rmse_stacked.sel(time=ds_rmse_stacked.dt.season==season)
                #df=ds_corr.to_dataframe(name='corr').reset_index() 
                #df=ds_rmse_stacked.to_dataframe().reset_index()
                df=ds_delta_rmse.to_dataframe().reset_index()
                lead_str=str(lead)+'_'+season+'_'+var+'_linear'
                #clim_ratio(df,[var])
                choropleth(df,lead_str,var, vmin=-0.24, vmax=0.24)
                


if __name__ == '__main__':
    main()
    