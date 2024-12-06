#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:58:05 2024

@author: amirouyed
Extract the numpy arrays from ds_swe and ds_sst, ds_sswt is input


"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import xarray as xr
import numpy as np
import pandas as pd

# Define the input shape of your images



def normalize_ds(ds):
    climatology_mean = ds.groupby("time.month").mean("time")
    anomalies = ds.groupby("time.month") - climatology_mean
    for var in ds: 
        ds[var+'_anom']=ds[var].groupby("time.month")-climatology_mean[var]
        ds[var+'_climatology']=climatology_mean[var]

    return ds


def initialize_model(input_shape):

# Initialize the CNN model
    model = Sequential()
    
    # Add convolutional layers
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the 2D feature maps into a 1D feature vector
    model.add(Flatten())
    
    # Add dense layers for regression
    model.add(Dense(128, activation='relu'))
    model.add(Dense(18, activation='linear'))  # Output layer with linear activation for regression
    
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    # Print the model summary
    model.summary()
    return model
    
def align_ds(ds,ds_sst, lead_month, month):
    #ds = ds.sel(time=ds.time.dt.month == 1)
    deltat = np.timedelta64(lead_month*30, 'D')
    times=ds['time'].values - deltat
    ds_sst = ds_sst.sel(time=times, method='nearest')
    return ds, ds_sst
    
ds_sst = xr.open_dataset('../data/raw/gistemp1200_GHCNv4_ERSSTv5.nc')
ds_sst = ds_sst.coarsen(lat=5, lon=5, boundary='trim').mean()


#input_shape = (image_height, image_width, num_channels




season = 'SON'

ds_swe=xr.open_dataset('../data/raw/PRISM_UA_data_broxon.nc')
breakpoint()
dates = pd.to_datetime( pd.DataFrame({'year': ds_swe['Year'].values, 'month': ds_swe['Month'].values, 'day':15}))
ds_swe=ds_swe.assign_coords(time=dates.values)
ds_swe=ds_swe.rename({'watershed':'basin'})
ds_swe=normalize_ds(ds_swe)
ds_swe, ds_sst=align_ds(ds_swe, ds_sst, 1, 1)


x=ds_sst['tempanomaly'].values
x=np.expand_dims(x, axis=3)
x=np.nan_to_num(x)
y=ds_swe['SWE_anom'].values
y=np.swapaxes(y,1,0)

breakpoint()
model=initialize_model(x.shape[1:])
model.fit(x, y, batch_size=32, epochs=100, validation_split=0.2)

#ds_swe['datetime']=(['time'], dates.values)
#ds_unit=align_ds(ds_swe,ds_sst, lat, lon, 1, season,basin, var_label)

breakpoint()