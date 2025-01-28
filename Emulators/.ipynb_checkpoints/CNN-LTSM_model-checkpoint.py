import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Input, Reshape, AveragePooling2D, MaxPooling2D, Conv2DTranspose, TimeDistributed, LSTM, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import (
    SpatialDropout2D, BatchNormalization
)

import random 

simus = ['ssp126',
         'ssp370',
         'ssp585',
         'hist-GHG',
         'hist-aer']

len_historical = 165

X_train = []
Y_train = []

# Run this py file in the same location as all of the nc files

for i, simu in enumerate(simus):

    input_name = 'inputs_' + simu + '.nc'
    output_name = 'outputs_' + simu + '.nc'

    # Just load hist data in these cases 'hist-GHG' and 'hist-aer'
    if 'hist' in simu:
        # load inputs 
        input_xr = xr.open_dataset(input_name)
            
        # load outputs                                                             
        output_xr = xr.open_dataset(output_name).mean(dim='member')
        output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                      "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                               'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])
    
    # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
    else:
        # load inputs 
        input_xr = xr.open_mfdataset(['inputs_historical.nc', 
                                    input_name]).compute()
            
        # load outputs                                                             
        output_xr = xr.concat([xr.open_dataset('outputs_historical.nc').mean(dim='member'),
                               xr.open_dataset(output_name).mean(dim='member')],
                               dim='time').compute()
        output_xr = output_xr.assign({"pr": output_xr.pr * 86400,
                                      "pr90": output_xr.pr90 * 86400}).rename({'lon':'longitude', 
                                                                               'lat': 'latitude'}).transpose('time','latitude', 'longitude').drop(['quantile'])

    print(input_xr.dims, simu)

    # Append to list 
    X_train.append(input_xr)
    Y_train.append(output_xr)

# Utilities for normalizing the input data
def normalize(data, var, meanstd_dict):
    mean = meanstd_dict[var][0]
    std = meanstd_dict[var][1]
    return (data - mean)/std

def unnormalize(data, var, meanstd_dict):
    mean = meanstd_dict[var][0]
    std = meanstd_dict[var][1]
    return data * std + mean

# Compute mean/std of each variable for the whole dataset
meanstd_inputs = {}

for var in ['CO2', 'CH4', 'SO2', 'BC']:
    # To not take the historical data into account several time we have to slice the scenario datasets
    # and only keep the historical data once (in the first ssp index 0 in the simus list)
    array = np.concatenate([X_train[i][var].data for i in [0, 3, 4]] + 
                           [X_train[i][var].sel(time=slice(len_historical, None)).data for i in range(1, 3)])
    print((array.mean(), array.std()))
    meanstd_inputs[var] = (array.mean(), array.std())

# normalize input data 
X_train_norm = [] 
for i, train_xr in enumerate(X_train): 
    for var in ['CO2', 'CH4', 'SO2', 'BC']: 
        var_dims = train_xr[var].dims
        train_xr=train_xr.assign({var: (var_dims, normalize(train_xr[var].data, var, meanstd_inputs))}) 
    X_train_norm.append(train_xr)

slider = 10 # years moving temporal window 

# Functions for reshaping the data 
def input_for_training(X_train_xr, skip_historical=False, len_historical=None): 
    
    X_train_np =  X_train_xr.to_array().transpose('time', 'latitude', 'longitude', 'variable').data

    time_length = X_train_np.shape[0]
    # If we skip historical data, the first sequence created has as last element the first scenario data point
    if skip_historical:
        X_train_to_return = np.array([X_train_np[i:i+slider] for i in range(len_historical-slider+1, time_length-slider+1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        X_train_to_return = np.array([X_train_np[i:i+slider] for i in range(0, time_length-slider+1)])
    
    return X_train_to_return 


def output_for_training(Y_train_xr, var, skip_historical=False, len_historical=None): 
    Y_train_np = Y_train_xr[var].data
    
    time_length = Y_train_np.shape[0]
    
    # If we skip historical data, the first sequence created has as target element the first scenario data point
    if skip_historical:
        Y_train_to_return = np.array([[Y_train_np[i+slider-1]] for i in range(len_historical-slider+1, time_length-slider+1)])
    # Else we just go through the whole dataset historical + scenario (does not matter in the case of 'hist-GHG' and 'hist_aer')
    else:
        Y_train_to_return = np.array([[Y_train_np[i+slider-1]] for i in range(0, time_length-slider+1)])
    
    return Y_train_to_return

# Building the models:

var_to_predict =  'tas'
# skip_historical set to (i < 2) because of the order of the scenario and historical runs in the X_train and Y_train lists.
# In details: ssp126 0, ssp370 1 = skip historical part of the data, ssp585 2, hist-GHG 3 and hist-aer 4 = keep the whole sequence
X_train_all = np.concatenate([input_for_training(X_train_norm[i], skip_historical=(i<2), len_historical=len_historical) for i in range(len(simus))], axis = 0)
Y_train_all = np.concatenate([output_for_training(Y_train[i], var_to_predict, skip_historical=(i<2), len_historical=len_historical) for i in range(len(simus))], axis=0)
print(X_train_all.shape) #(726, 10, 96, 144, 4)
print(Y_train_all.shape) #(726, 1, 96, 144)

seed = 6 
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

keras.backend.clear_session()
cnn_model = None

# Uncomment when you want specific model

# # ADAM
# cnn_model = Sequential()
# cnn_model.add(Input(shape=(slider, 96, 144, 4)))
# cnn_model.add(TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='relu'), input_shape=(slider, 96, 144, 4)))
# cnn_model.add(TimeDistributed(AveragePooling2D(2)))
# cnn_model.add(TimeDistributed(GlobalAveragePooling2D()))
# cnn_model.add(LSTM(25, activation='relu'))
# cnn_model.add(Dense(1*96*144))
# cnn_model.add(Activation('linear'))
# cnn_model.add(Reshape((1, 96, 144)))

# # Initialize the ADAM optimizer
# adam_optimizer = Adam(learning_rate=0.001)  # You can adjust the learning rate if needed

# # Compile the model with the ADAM optimizer
# cnn_model.compile(optimizer=adam_optimizer, 
#                   loss='mean_squared_error',  # Specify the loss function (e.g., MSE for regression tasks)
#                   metrics=['mse'])  # Add any metrics you'd like to track

# --------------------------------------------------------------------------------------------------------------------------------------

# # LTSM Regularizer
# cnn_model = Sequential()
# cnn_model.add(Input(shape=(slider, 96, 144, 4)))
# cnn_model.add(TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='relu'), input_shape=(slider, 96, 144, 4)))
# cnn_model.add(TimeDistributed(AveragePooling2D(2)))
# cnn_model.add(TimeDistributed(GlobalAveragePooling2D()))
# cnn_model.add(LSTM(25, activation='relu'))

# # Added kernel regularizer
# cnn_model.add(Dense(1 * 96 * 144, kernel_regularizer=l2(0.01)))

# cnn_model.add(Activation('linear'))
# cnn_model.add(Reshape((1, 96, 144)))

# cnn_model.compile(optimizer='rmsprop', loss='mse', metrics=['mse']) # Default optimizer

# --------------------------------------------------------------------------------------------------------------------------------------

# Spartial 2D Dropout

cnn_model = Sequential()
cnn_model.add(Input(shape=(slider, 96, 144, 4)))
cnn_model.add(TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='relu'), input_shape=(slider, 96, 144, 4)))
cnn_model.add(TimeDistributed(AveragePooling2D(2)))
cnn_model.add(TimeDistributed(GlobalAveragePooling2D()))

cnn_model.add(TimeDistributed(SpatialDropout2D(0.2)))  # Adding spatial dropout layer
cnn_model.add(TimeDistributed(BatchNormalization()))   # Adding batch normalization layer

cnn_model.add(LSTM(25, activation='relu', return_sequences=False))

cnn_model.add(Activation('linear'))
cnn_model.add(Reshape((1, 96, 144)))

cnn_model.compile(optimizer='rmsprop', loss='mse', metrics=['mse']) # Default optimizer

# --------------------------------------------------------------------------------------------------------------------------------------

# Summary to confirm model structure
cnn_model.summary()

# Training

hist = cnn_model.fit(X_train_all,
                     Y_train_all,
                     # use_multiprocessing=True, 
                     #workers=5,
                     batch_size=16,
                     epochs=30,
                     verbose=1)

# Open and reformat test data 
X_test = xr.open_mfdataset(['inputs_historical.nc',
                            'inputs_ssp245.nc']).compute()

# Normalize data 
for var in ['CO2', 'CH4', 'SO2', 'BC']: 
    var_dims = X_test[var].dims
    X_test = X_test.assign({var: (var_dims, normalize(X_test[var].data, var, meanstd_inputs))}) 
    
X_test_np = input_for_training(X_test, skip_historical=False, len_historical=len_historical)  

vars_to_predict = ['tas', 'diurnal_temperature_range', 'pr', 'pr90']

# Open and reformat test data 
X_test = xr.open_mfdataset(['inputs_historical.nc',
                             'inputs_ssp245.nc']).compute()

# Normalize input data 
for var in ['CO2', 'CH4', 'SO2', 'BC']: 
    var_dims = X_test[var].dims
    X_test = X_test.assign({var: (var_dims, normalize(X_test[var].data, var, meanstd_inputs))}) 
    
X_test_np = input_for_training(X_test, skip_historical=False, len_historical=len_historical) 

# Predictions

for var_to_predict in vars_to_predict:
    
    print(var_to_predict)
    
    # Data
    X_train_all = np.concatenate([input_for_training(X_train_norm[i], skip_historical=(i<2), len_historical=len_historical) for i in range(len(simus))], axis = 0)
    Y_train_all = np.concatenate([output_for_training(Y_train[i], var_to_predict, skip_historical=(i<2), len_historical=len_historical) for i in range(len(simus))], axis=0)
    print(X_train_all.shape)
    print(Y_train_all.shape)
    
    # Make predictions using trained model 
    m_pred = cnn_model.predict(X_test_np)
    # Reshape to xarray 
    m_pred = m_pred.reshape(m_pred.shape[0], m_pred.shape[2], m_pred.shape[3])
    m_pred = xr.DataArray(m_pred, dims=['time', 'lat', 'lon'], coords=[X_test.time.data[slider-1:], X_test.latitude.data, X_test.longitude.data])
    xr_prediction = m_pred.transpose('lat', 'lon', 'time').sel(time=slice(2015,2101)).to_dataset(name=var_to_predict)

    if var_to_predict=="pr90" or var_to_predict=="pr":
        xr_prediction = xr_prediction.assign({var_to_predict: xr_prediction[var_to_predict] / 86400})

    # Save test predictions as .nc 
    if var_to_predict == 'diurnal_temperature_range':
        xr_prediction.to_netcdf('outputs_ssp245_predict_dtr.nc', 'w')
    else:
        xr_prediction.to_netcdf('outputs_ssp245_predict_{}.nc'.format(var_to_predict), 'w')
    xr_prediction.close()
