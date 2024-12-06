#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:57:15 2023

@author: amirouyed

read van del doool on teleconnections
"""
import xarray as xr

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader

BASIN=12

def map_plotter_cartopy(ds, title, label, cmap, units_label=''):
    values = np.squeeze(ds[label].values)
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines()
    im = ax.imshow(values, cmap=cmap, extent=[ds['lon'].min(
    ), ds['lon'].max(), ds['lat'].min(), ds['lat'].max()], vmin=0,vmax=0.2,origin='lower')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    plt.title(label)
    cbar_ax = fig.add_axes([0.1, 0.05, 0.78, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
                 pad=0.5, label=units_label)
   # plt.savefig('../data/processed/plots/'+title+'.png',
    #            bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    

    



def median_r2(basins):
    d = {'season': [], 'basin':[],'r2_median': []}
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        ds = xr.open_dataset('../data/processed/ds_corr_' + season+'.nc')
        ds['r2'] = ds['corr']**2
        
        for basin in basins:
            ds_unit=ds.sel(basin=basin)
            r2_median = ds_unit['r2'].median().item()
            d['season'].append(season)
            d['basin'].append(basin)
            d['r2_median'].append(r2_median)
    df = pd.DataFrame(data=d)
    print(df.loc[df['r2_median'].idxmax()])
    
def choropleth(df, var, vmin=-0.2, vmax=0.2, cmap_l='RdBu'):
    indexes=df['basin'].values
    index_str=[f"{i:02}" for i in indexes]
    df['basin_str']=np.array(index_str)
    fig=plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    ax.set_extent([-130, -60, 20, 55], ccrs.PlateCarree())
    ax.set_title('temp (2m) anomaly')
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
    fig.colorbar(sm, ax=ax, cax=cbar_ax,orientation="horizontal", pad=0.3, label='Δ r2')
    #fig.colorbar(ax, cax=cbar_ax,orientation="horizontal", pad=0.5, label=units_label) 
    fig.tight_layout(rect=[0, 1.3, 1, 1.2])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    plt.savefig('../data/processed/plots/r2.png', dpi=300)
    plt.show()
    plt.close()


    
def map_plotter_ax(ax,ds,label, basin,cmap, vmin, vmax ):
    values = np.squeeze(ds[label].values)
    ax.coastlines()
    ax.gridlines()
    im = ax.imshow(values, cmap=cmap, extent=[ds['lon'].min(
    ), ds['lon'].max(), ds['lat'].min(), ds['lat'].max()], vmin=vmin,vmax=vmax,origin='lower')
    ax.gridlines(draw_labels=False, x_inline=False, y_inline=False)
    ax.set_title('basin: '+str(basin))
    return ax, im
   

def four_panels_plot(ds, label, basins, cmap, vmin, vmax):
    fig, axes=plt.subplots(nrows=2, ncols=2, layout='compressed', subplot_kw={
                             'projection': ccrs.PlateCarree()})
    axlist=axes.flatten()
    for i, basin in enumerate(basins):
        ds_unit=ds.sel(basin=basin)
        axlist[i], im= map_plotter_ax(axlist[i],ds_unit,label, basin , cmap, vmin, vmax)
        
    
    
    #plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist())

    basin_range=str(basins[0])+'-'+str(basins[-1])
    plt.savefig('../data/processed/plots/four_panel_'+label+'_basins'+basin_range+'.png',bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()



def plotting_pipeline(ds_plot, var_plot, cmap, vmin, vmax):
    four_panels_plot(ds_plot,var_plot, [1,2,3,4],cmap,vmin, vmax)
    four_panels_plot(ds_plot,var_plot, [5,6,7,8],cmap, vmin, vmax)
    four_panels_plot(ds_plot, var_plot, [9,10,11,12],cmap, vmin, vmax)
    four_panels_plot(ds_plot,var_plot, [13,14,15,16],cmap, vmin, vmax)
    four_panels_plot(ds_plot, var_plot, [14,15,16,17],cmap, vmin, vmax)


ds1 = xr.open_dataset('../data/processed/ds_corr_DJF_lead_1.nc')
ds1['r2'] = ds1['corr']**2


ds2= xr.open_dataset('../data/processed/ds_corr_TMean_DJF_lead_1.nc')
ds2['r2'] = ds2['corr']**2
ds1['delta_r2_TMean']=ds1['r2']-ds2['r2']

ds2= xr.open_dataset('../data/processed/ds_corr_PPT_DJF_lead_1.nc')
ds2['r2'] = ds2['corr']**2
ds1['delta_r2_PPT']=ds1['r2']-ds2['r2']

var_plot='delta_r2'
cmap='RdBu'
ds_plot=ds1.copy()
vmin=-0.1
vmax=0.1
plotting_pipeline(ds1, 'r2', 'viridis_r', 0, 0.3)

plotting_pipeline(ds1, 'delta_r2_TMean', cmap, -0.1, 0.1)
plotting_pipeline(ds1, 'delta_r2_PPT', cmap, -0.1, 0.1)

