'''Import Statements'''

import numpy as np
import pandas as pd
import xarray as xr
import zipfile
import os
from sklearn.linear_model import LinearRegression 

'''Utilities for Normalizing the Emissions Data'''

min_co2 = 0.
max_co2 = 2400
def normalize_co2(data):
    return data / max_co2

def un_normalize_co2(data):
    return data * max_co2

min_ch4 = 0.
max_ch4 = 0.6
def normalize_ch4(data):
    return data / max_ch4

def un_normalize_ch4(data):
    return data * max_ch4

data_path = './ClimateBench/'


'''Pattern Scaling Model'''

# Get one combined historical + ssp585 + ssp126 + ssp370 timeseries for now
X = xr.concat([xr.open_dataset(data_path + 'inputs_historical.nc'), xr.open_dataset(data_path + 'inputs_ssp585.nc'),xr.open_dataset(data_path+ 'inputs_ssp126.nc'),xr.open_dataset(data_path+ 'inputs_ssp370.nc')], dim='time').compute()

# Take the 2nd ensemble member for the historical (the first one has some missing DTR values for some reason...) and the 1st one for ssp585, ssp126, ssp370
Y = xr.concat([xr.open_dataset(data_path + 'outputs_historical.nc').sel(member=2), xr.open_dataset(data_path + 'outputs_ssp585.nc').sel(member=1),xr.open_dataset(data_path+ 'outputs_ssp126.nc').sel(member=1),xr.open_dataset(data_path+ 'outputs_ssp370.nc').sel(member=1)], dim='time').compute()

# Convert the precip values to mm/day
Y["pr"] *= 86400
Y["pr90"] *= 86400

X["time" ]= np.arange(1,424) 
Y["time"] = np.arange(1,424)

# TAS in historical, SSP126, SSP370, SSP585.
y_inp_tas = Y["tas"].stack(dims=["lat", "lon"])

# SSP245
test_Y = xr.open_dataset(data_path + 'outputs_ssp245.nc').compute()
test_X = xr.open_dataset(data_path + 'inputs_ssp245.nc').compute()

test_inputs = pd.DataFrame({
    "CO2": normalize_co2(test_X["CO2"].data),
}, index=test_X["CO2"].coords['time'].data)


ps_tas = LinearRegression(fit_intercept=False)
ps_tas.fit(X['CO2'].to_numpy().reshape(-1, 1), y_inp_tas)

pred_tas = ps_tas.predict(test_X['CO2'].to_numpy().reshape(-1, 1)).reshape(86, 96, 144)

true_tas = test_Y["tas"].mean('member')


'''Taking in ClimateBench Emulator Outputs for Custom Scenarios'''

possible_carbons = np.arange(0, 9510, 10)

last_hist_CO2 = xr.open_dataset(data_path + 'inputs_historical.nc')['CO2'].data[-1]

def create_carbon_preds(possible_carbons):

    for carbon in possible_carbons:
        
        end_CO2 = carbon
        custom_data = np.linspace(last_hist_CO2, end_CO2, num=86).reshape(-1, 1)
        m_out_tas = ps_tas.predict(custom_data).reshape(86, 96, 144)

        xr_custom = xr.Dataset(coords={'time': test_X.time.values, 'lat': test_X.latitude.values, 'lon': test_X.longitude.values})
        xr_custom["tas"] = (['time', 'lat', 'lon'],  m_out_tas)
        X_custom = xr_custom['tas'].weighted(weights).mean(['lat', 'lon']).to_pandas().values.reshape(-1, 1)

        SLR_custom = pd.DataFrame({'year': np.arange(2015, 2101)})
        
        SLR_custom['5q_dH_dT'] = model_5q.predict(X_custom) ### IMPORTANT... if we don't want to go through the trouble 
        SLR_custom['17q_dH_dT'] = model_17q.predict(X_custom) ### of training each model, we can save the model weights
        SLR_custom['50q_dH_dT'] = model_50q.predict(X_custom) ### (intercept & slope) and just save those in a file and 
        SLR_custom['83q_dH_dT'] = model_83q.predict(X_custom) ### do the calculation manually, to make the app more 
        SLR_custom['95q_dH_dT'] = model_95q.predict(X_custom) ### more efficient.

        SLR_custom = SLR_custom.set_index('year').cumsum() * 1000 #<- if want in mm, otherwise remove.

        SLR_custom.to_csv(f"PS_Carbon/PS_Carbon_{carbon}_Preds.csv")

create_carbon_preds(possible_carbons)

folder_to_zip = "PS_Carbon"
zip_filename = "PS_Carbon.zip"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(folder_to_zip):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, folder_to_zip))
