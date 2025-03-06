import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof
import esem

import warnings
warnings.filterwarnings('ignore')
from esem import gp_model
from esem.data_processors import Whiten, Normalise
from tensorflow.keras.models import load_model

from SLR import model_5q, model_17q, model_50q, model_83q, model_95q

data_path_1 = "data/inputs_outputs/"

data_path_2 = "data/"

min_co2 = 0.
max_co2 = 9500
def normalize_co2(data):
    return data / max_co2

def un_normalize_co2(data):
    return data * max_co2

min_ch4 = 0.
max_ch4 = 0.8
def normalize_ch4(data):
    return data / max_ch4

def un_normalize_ch4(data):
    return data * max_ch4

# Load the saved model in .keras format
cnn_model = load_model(data_path_2 + "cnn_model_default.keras")

simus = ['ssp126',
         'ssp370',
         'ssp585',
         'hist-GHG',
         'hist-aer']

len_historical = 165

X_train = []
Y_train = []

for i, simu in enumerate(simus):

    input_name = data_path_1 + 'inputs_' + simu + '.nc'
    output_name = data_path_1 + 'outputs_' + simu + '.nc'

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
        input_xr = xr.open_mfdataset([data_path_1 + 'inputs_historical.nc', 
                                    input_name]).compute()
            
        # load outputs                                                             
        output_xr = xr.concat([xr.open_dataset(data_path_1 + 'outputs_historical.nc').mean(dim='member'),
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
    # print((array.mean(), array.std()))
    meanstd_inputs[var] = (array.mean(), array.std())

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

# Open and reformat test data 
X_test = xr.open_mfdataset([data_path_1 + 'inputs_ssp245.nc']).compute()

# Normalize data 
for var in ['CO2', 'CH4', 'SO2', 'BC']: 
    var_dims = X_test[var].dims
    X_test = X_test.assign({var: (var_dims, normalize(X_test[var].data, var, meanstd_inputs))}) 
    
X_test_np = input_for_training(X_test, skip_historical=False, len_historical=len_historical)  

X_test = xr.open_mfdataset([data_path_1 + 'inputs_historical.nc',
                            data_path_1 + 'inputs_ssp245.nc']).compute()

# X_test = X_test.sel(time=slice("2015", "2100"))
X_test_2025 = X_test.where(X_test["time"] == 2025, drop=True)
SO2_2025 = X_test_2025["SO2"].to_numpy()
CH4_2025 = X_test_2025["CH4"].to_numpy()
BC_2025 = X_test_2025["BC"].to_numpy()

possible_carbons = np.arange(0, 9510, 10)
# possible_carbons = np.array([4520])

def create_cnn_carbon_preds(possible_carbons):

    last_hist_CO2 = xr.open_dataset(data_path_1 + 'inputs_historical.nc')['CO2'].data[-1]
    
    for carbon in possible_carbons:
        
        # Instead of just linearly of 2015-2100, this increases CO2 linearly including historical
        step = normalize_co2(np.linspace(last_hist_CO2, carbon, 251))

        X_test["CO2"][:] = step
        X_test["SO2"][:] = SO2_2025
        X_test["CH4"][:] = CH4_2025
        X_test["BC"][:] = BC_2025

        # return X_test
            
        X_test_np = input_for_training(X_test, skip_historical=False, len_historical=len_historical) 


        # Make predictions using trained model 
        m_pred = cnn_model.predict(X_test_np)
        # reshape to xarray 
        m_pred = m_pred.reshape(m_pred.shape[0], m_pred.shape[2], m_pred.shape[3])
        m_pred = xr.DataArray(m_pred, dims=['time', 'lat', 'lon'], coords=[X_test.time.data[slider-1:], X_test.latitude.data, X_test.longitude.data])
        m_pred = m_pred.transpose('lat', 'lon', 'time').sel(time=slice(2015, 2101)).to_dataset(name="tas")
        # return m_pred
        cnn = m_pred.mean(dim=("lat", "lon"))
        X_custom = cnn["tas"].values.reshape(-1, 1)
        
        SLR_custom = pd.DataFrame({'year': np.arange(2015, 2101)})
        
        SLR_custom['5q_dH_dT'] = model_5q.predict(X_custom) ### IMPORTANT... if we don't want to go through the trouble 
        SLR_custom['17q_dH_dT'] = model_17q.predict(X_custom) ### of training each model, we can save the model weights
        SLR_custom['50q_dH_dT'] = model_50q.predict(X_custom) ### (intercept & slope) and just save those in a file and 
        SLR_custom['83q_dH_dT'] = model_83q.predict(X_custom) ### do the calculation manually, to make the app more 
        SLR_custom['95q_dH_dT'] = model_95q.predict(X_custom) ### more efficient.
        
        SLR_custom = SLR_custom.set_index('year').cumsum() * 1000 #<- if want in mm, otherwise remove.

        # Uncomment line below and create appropriate CNN_245_Linear folder to save multiple CSVs
        # SLR_custom.to_csv(f"data/CNN_245_Linear/CNN_Carbon_{carbon}_Preds.csv")

        if carbon == 4520:
            SLR_custom.to_csv(f"data/CNN_Carbon_{carbon}_Preds.csv")
            print("CNN model for sea level rise using SSP 245 with 4520 gigatons of cumulative carbon dioxide csv has been created!")

create_cnn_carbon_preds(possible_carbons)