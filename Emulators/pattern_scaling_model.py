import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import LinearRegression 
from glob import glob

data_path = "ClimateBench/" # Change to your working directory.

# Path to save the netcdf file.
path_output = data_path + 'outputs_ssp245_prediction_pattern2.nc'

# Path to testing data (ssp245).
pathssp245_Y = 'outputs_ssp245.nc'
pathssp245_X = 'inputs_ssp245.nc'

# Utilities for normalizing the emissions data.
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

# data_path = "./"

from glob import glob

inputs = glob(data_path + "inputs_s*.nc")
SECONDS_IN_YEAR = 60*60*24*365 #s

fig, axes = plt.subplots(2, 2, figsize=(12,12))

for input in inputs:
    label=input.split('_')[1][:-3]
    X = xr.open_dataset(input)
    x = range(2015, 2101)

    weights = np.cos(np.deg2rad(X.latitude))
    
    axes[0, 0].plot(x, X['CO2'].data, label=label)
    axes[0, 0].set_ylabel(
        "Cumulative anthropogenic CO2 \nemissions since 1850 (GtCO2)")
    axes[0, 1].plot(x, X['CH4'].data, label=label)
    axes[0, 1].set_ylabel("Anthropogenic CH4 \nemissions (GtCH4 / year)")
    # FIXME: Not sure where this factor of 1000 comes from...! 
    # Maybe the CEDS data is really g/m-2/s?
    axes[1, 0].plot(x, X['SO2'].weighted(weights).sum(
        ['latitude', 'longitude']).data*SECONDS_IN_YEAR*1e-9, label=label)
    axes[1, 0].set_ylabel("Anthropogenic SO2 \nemissions (GtSO2 / year)")
    axes[1, 1].plot(x, X['BC'].weighted(weights).sum(
        ['latitude', 'longitude']).data*SECONDS_IN_YEAR*1e-9, label=label)
    axes[1, 1].set_ylabel("Anthropogenic BC \nemissions (GtBC / year)")

axes[0, 0].set_title('CO2')
axes[0, 1].set_title('CH4')
axes[1, 0].set_title('SO2')
axes[1, 1].set_title('BC')
axes[0, 0].legend()
plt.tight_layout()

# Get one combined historical + ssp585 + ssp126 + ssp370 timeseries for now.
X = xr.concat([
    xr.open_dataset(data_path + 'inputs_historical.nc'), 
    xr.open_dataset(data_path + 'inputs_ssp585.nc'),
    xr.open_dataset(data_path + 'inputs_ssp126.nc'),
    xr.open_dataset(data_path + 'inputs_ssp370.nc')], dim='time').compute()

# Take the 2nd ensemble member for the historical (the first one has some 
# missing DTR values for some reason...) and the 1st one for ssp585, ssp126, 
# ssp370.
Y = xr.concat([
    xr.open_dataset(data_path + 'outputs_historical.nc').sel(member=2), 
    xr.open_dataset(data_path + 'outputs_ssp585.nc').sel(member=1),
    xr.open_dataset(data_path + 'outputs_ssp126.nc').sel(member=1),
    xr.open_dataset(data_path + 'outputs_ssp370.nc').sel(member=1)], 
    dim='time').compute()

# Convert the precip values to mm/day.
Y["pr"] *= 86400
Y["pr90"] *= 86400

X["time"]=np.arange(1,424) 
Y["time"]=np.arange(1,424)

weights = np.cos(np.deg2rad(Y.lat))
global_mean_temp = Y['tas'].weighted(weights).mean(['lat', 'lon']).to_pandas()

def get_rmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.lat))
    return np.sqrt(((truth-pred)**2).weighted(weights).mean(
        ['lat', 'lon'])).data.mean()

y_inp_tas=Y["tas"].stack(dims=["lat", "lon"])
y_inp_pr=Y["pr"].stack(dims=["lat", "lon"])
y_inp_pr90=Y["pr90"].stack(dims=["lat", "lon"])
y_inp_dtr=Y["diurnal_temperature_range"].stack(dims=["lat", "lon"])


'''Pattern Scaling Model'''

# TAS in historical, SSP126, SSP370, SSP585.
y_inp_tas = Y["tas"].stack(dims=["lat", "lon"])

# SSP245
test_Y = xr.open_dataset('./ClimateBench/outputs_ssp245.nc').compute()
test_X = xr.open_dataset('./ClimateBench/inputs_ssp245.nc').compute()

test_inputs = pd.DataFrame(
    {"CO2": normalize_co2(test_X["CO2"].data),}, 
    index=test_X["CO2"].coords['time'].data)

ps_tas = LinearRegression(fit_intercept=False)
ps_tas.fit(X['CO2'].to_numpy().reshape(-1, 1), y_inp_tas)

pred_tas = ps_tas.predict(
    test_X['CO2'].to_numpy().reshape(-1, 1)).reshape(86, 96, 144)

true_tas = test_Y["tas"].mean('member')

print(f"RMSE: {get_rmse(true_tas[65:], pred_tas[65:])}")
print(f"RMSE: {get_rmse(true_tas[35:], pred_tas[35:])}")

# The model is ps_tas, and takes in CO2 levels as input.
