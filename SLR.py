'''Import Statements'''

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression 
from glob import glob


'''Process Emulator Inputs'''

# Change this path to where your emulator outputs are stored. We are training
# the SLR model on historical data and predicting on SSP data.
data_path = "ClimateBench/"

inputs = glob(data_path + "inputs_s*.nc")
SECONDS_IN_YEAR = 60*60*24*365 #s

for input in inputs:
    label=input.split('_')[1][:-3]
    X = xr.open_dataset(input)
    x = range(2015, 2101)

    # Need this block and the next for the weights.
    weights = np.cos(np.deg2rad(X.latitude))


'''More Input Processing'''

# Get one combined historical + ssp585 + ssp126 + ssp370 timeseries for now.
X = xr.concat([
    xr.open_dataset(data_path + 'inputs_historical.nc'), 
    xr.open_dataset(data_path + 'inputs_ssp585.nc'),
    xr.open_dataset(data_path + 'inputs_ssp126.nc'),
    xr.open_dataset(data_path + 'inputs_ssp370.nc')], 
    dim='time').compute()

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

# Need the weights here as well. Very important in order to weight
# the output file TAS variables.
weights = np.cos(np.deg2rad(Y.lat))
global_mean_temp = Y['tas'].weighted(weights).mean(['lat', 'lon']).to_pandas()

historical_tas = xr.open_dataset(data_path 
+ "outputs_historical.nc")["tas"].weighted(weights).mean(
    ['lat', 'lon', 'member']).to_pandas()
ssp126_tas = xr.open_dataset(data_path 
+ "outputs_ssp126.nc")["tas"].weighted(weights).mean(
    ['lat', 'lon', 'member']).to_pandas()
ssp245_tas = xr.open_dataset(data_path 
+ "outputs_ssp245.nc")["tas"].weighted(weights).mean(
    ['lat', 'lon', 'member']).to_pandas()
ssp370_tas = xr.open_dataset(data_path 
+ "outputs_ssp370.nc")["tas"].weighted(weights).mean(
    ['lat', 'lon', 'member']).to_pandas()
ssp585_tas = xr.open_dataset(data_path 
+ "outputs_ssp585.nc")["tas"].weighted(weights).mean(
    ['lat', 'lon', 'member']).to_pandas()


'''Plot of Historical Sea Level'''

# Read in SLR data.
more_sl = pd.read_excel("data/global_basin_timeseries.xlsx", index_col=0)
gmsl = more_sl[["Observed GMSL [mean]"]]
gmsl_anom = gmsl - gmsl.loc[1900]


'''Fitting Rahmstorf 2007 Model on Historical'''

# Training historical data.
X_hist = historical_tas.loc[1901:2014].values.reshape(-1, 1)
y = gmsl_anom.diff().loc[1901:2014].values.reshape(-1, 1)

# Fit the linear regression model.
hist_model = LinearRegression()
hist_model.fit(X_hist, y)

# Predict and plot data.
hist_sl = pd.DataFrame({"year": range(1901, 2015), 
"pred dH_dt (mm/yr)": hist_model.predict(X_hist).flatten()})
hist_sl = hist_sl.set_index("year")
hist_sl = hist_sl.join(gmsl_anom, how='left')
hist_sl['pred_sea_level_rise'] = hist_sl['pred dH_dt (mm/yr)'].cumsum()

# MSE for historical model performance.
MSE_TAS = sum((hist_sl['Observed GMSL [mean]'] 
- hist_sl['pred_sea_level_rise']) ** 2) / hist_sl.shape[0]


'''Fitting Rahmstorf Model on SSP Data'''

# Features (temperature anomaly) and target (rate of sea level rise).
X_ssp126 = ssp126_tas.values.reshape(-1, 1)
X_ssp245 = ssp245_tas.values.reshape(-1, 1)
X_ssp370 = ssp370_tas.values.reshape(-1, 1)
X_ssp585 = ssp585_tas.values.reshape(-1, 1)

# ClimateBench output TAS.
df_126_tas = pd.DataFrame(ssp126_tas).reset_index().rename(
    columns={'time': 'year', 0: 'TAS'})
df_245_tas = pd.DataFrame(ssp245_tas).reset_index().rename(
    columns={'time': 'year', 0: 'TAS'})
df_370_tas = pd.DataFrame(ssp370_tas).reset_index().rename(
    columns={'time': 'year', 0: 'TAS'})
df_585_tas = pd.DataFrame(ssp585_tas).reset_index().rename(
    columns={'time': 'year', 0: 'TAS'})


'''5th Quantile Model'''

# 5th quantile.
data_5 = pd.read_csv('SLR_Training/5_quantile_yearly.csv')
data_5_ssp126 = df_126_tas.merge(data_5[['year', 'ssp126']]).rename(
    columns={'ssp126': 'GMSLR'})
data_5_ssp245 = df_245_tas.merge(data_5[['year', 'ssp245']]).rename(
    columns={'ssp245': 'GMSLR'})
data_5_ssp370 = df_370_tas.merge(data_5[['year', 'ssp370']]).rename(
    columns={'ssp370': 'GMSLR'})
data_5_ssp585 = df_585_tas.merge(data_5[['year', 'ssp585']]).rename(
    columns={'ssp585': 'GMSLR'})

# Training data.
data_5_train = pd.concat([data_5_ssp126, 
                           data_5_ssp370, 
                           data_5_ssp585], 
                          axis=0, 
                          ignore_index=True)
X_train_5 = data_5_train['TAS'].values.reshape(-1, 1)
y_train_5 = data_5_train['GMSLR'].values.reshape(-1, 1)

# Validation (SSP245) data.
X_val_5 = data_5_ssp245['TAS'].values.reshape(-1, 1)

# Model for 5th quantile.
model_5q = LinearRegression()
model_5q.fit(X_train_5, y_train_5)

val_5q = data_5_ssp245.set_index('year')
val_5q['pred_dH_dT'] = model_5q.predict(X_val_5)
val_5q['expected_SL'] = val_5q['GMSLR'].cumsum() * 1000
val_5q['predicted_SL'] = val_5q['pred_dH_dT'].cumsum() * 1000


'''17th Quantile Model'''

# 17th quantile.
data_17 = pd.read_csv('SLR_Training/17_quantile_yearly.csv')
data_17_ssp126 = df_126_tas.merge(data_17[['year', 'ssp126']]).rename(
    columns={'ssp126': 'GMSLR'})
data_17_ssp245 = df_245_tas.merge(data_17[['year', 'ssp245']]).rename(
    columns={'ssp245': 'GMSLR'})
data_17_ssp370 = df_370_tas.merge(data_17[['year', 'ssp370']]).rename(
    columns={'ssp370': 'GMSLR'})
data_17_ssp585 = df_585_tas.merge(data_17[['year', 'ssp585']]).rename(
    columns={'ssp585': 'GMSLR'})

# Training data.
data_17_train = pd.concat([data_17_ssp126, 
                           data_17_ssp370, 
                           data_17_ssp585], 
                          axis=0, 
                          ignore_index=True)
X_train_17 = data_17_train['TAS'].values.reshape(-1, 1)
y_train_17 = data_17_train['GMSLR'].values.reshape(-1, 1)

# Validation (SSP245) data.
X_val_17 = data_17_ssp245['TAS'].values.reshape(-1, 1)

# Model for 17th quantile.
model_17q = LinearRegression()
model_17q.fit(X_train_17, y_train_17)

val_17q = data_17_ssp245.set_index('year')
val_17q['pred_dH_dT'] = model_17q.predict(X_val_17)
val_17q['expected_SL'] = val_17q['GMSLR'].cumsum() * 1000
val_17q['predicted_SL'] = val_17q['pred_dH_dT'].cumsum() * 1000


'''50th Quantile Model'''

# 50th quantile.
data_50 = pd.read_csv('SLR_Training/50_quantile_yearly.csv')
data_50_ssp126 = df_126_tas.merge(data_50[['year', 'ssp126']]).rename(
    columns={'ssp126': 'GMSLR'})
data_50_ssp245 = df_245_tas.merge(data_50[['year', 'ssp245']]).rename(
    columns={'ssp245': 'GMSLR'})
data_50_ssp370 = df_370_tas.merge(data_50[['year', 'ssp370']]).rename(
    columns={'ssp370': 'GMSLR'})
data_50_ssp585 = df_585_tas.merge(data_50[['year', 'ssp585']]).rename(
    columns={'ssp585': 'GMSLR'})

# Training data.
data_50_train = pd.concat([data_50_ssp126, 
                           data_50_ssp370, 
                           data_50_ssp585], 
                          axis=0, 
                          ignore_index=True)
X_train_50 = data_50_train['TAS'].values.reshape(-1, 1)
y_train_50 = data_50_train['GMSLR'].values.reshape(-1, 1)

# Validation (SSP245) data.
X_val_50 = data_50_ssp245['TAS'].values.reshape(-1, 1)

# Model for 50th quantile.
model_50q = LinearRegression()
model_50q.fit(X_train_50, y_train_50)

val_50q = data_50_ssp245.set_index('year')
val_50q['pred_dH_dT'] = model_50q.predict(X_val_50)
val_50q['expected_SL'] = val_50q['GMSLR'].cumsum() * 1000
val_50q['predicted_SL'] = val_50q['pred_dH_dT'].cumsum() * 1000


'''83rd Quantile Model'''

# 83rd quantile.
data_83 = pd.read_csv('SLR_Training/83_quantile_yearly.csv')
data_83_ssp126 = df_126_tas.merge(data_83[['year', 'ssp126']]).rename(
    columns={'ssp126': 'GMSLR'})
data_83_ssp245 = df_245_tas.merge(data_83[['year', 'ssp245']]).rename(
    columns={'ssp245': 'GMSLR'})
data_83_ssp370 = df_370_tas.merge(data_83[['year', 'ssp370']]).rename(
    columns={'ssp370': 'GMSLR'})
data_83_ssp585 = df_585_tas.merge(data_83[['year', 'ssp585']]).rename(
    columns={'ssp585': 'GMSLR'})

# Training data.
data_83_train = pd.concat([data_83_ssp126, 
                           data_83_ssp370, 
                           data_83_ssp585], 
                          axis=0, 
                          ignore_index=True)
X_train_83 = data_83_train['TAS'].values.reshape(-1, 1)
y_train_83 = data_83_train['GMSLR'].values.reshape(-1, 1)

# Validation (SSP245) data.
X_val_83 = data_83_ssp245['TAS'].values.reshape(-1, 1)

# Model for 83th quantile.
model_83q = LinearRegression()
model_83q.fit(X_train_83, y_train_83)

val_83q = data_83_ssp245.set_index('year')
val_83q['pred_dH_dT'] = model_83q.predict(X_val_83)
val_83q['expected_SL'] = val_83q['GMSLR'].cumsum() * 1000
val_83q['predicted_SL'] = val_83q['pred_dH_dT'].cumsum() * 1000


'''95th Quantile Model'''

# 95th quantile.
data_95 = pd.read_csv('SLR_Training/95_quantile_yearly.csv')
data_95_ssp126 = df_126_tas.merge(data_95[['year', 'ssp126']]).rename(
    columns={'ssp126': 'GMSLR'})
data_95_ssp245 = df_245_tas.merge(data_95[['year', 'ssp245']]).rename(
    columns={'ssp245': 'GMSLR'})
data_95_ssp370 = df_370_tas.merge(data_95[['year', 'ssp370']]).rename(
    columns={'ssp370': 'GMSLR'})
data_95_ssp585 = df_585_tas.merge(data_95[['year', 'ssp585']]).rename(
    columns={'ssp585': 'GMSLR'})

# Training data.
data_95_train = pd.concat([data_95_ssp126, 
                           data_95_ssp370, 
                           data_95_ssp585], 
                          axis=0, 
                          ignore_index=True)
X_train_95 = data_95_train['TAS'].values.reshape(-1, 1)
y_train_95 = data_95_train['GMSLR'].values.reshape(-1, 1)

# Validation (SSP245) data.
X_val_95 = data_95_ssp245['TAS'].values.reshape(-1, 1)

# Model for 95th quantile.
model_95q = LinearRegression()
model_95q.fit(X_train_95, y_train_95)

val_95q = data_95_ssp245.set_index('year')
val_95q['pred_dH_dT'] = model_95q.predict(X_val_95)
val_95q['expected_SL'] = val_95q['GMSLR'].cumsum() * 1000
val_95q['predicted_SL'] = val_95q['pred_dH_dT'].cumsum() * 1000


'''Function to Predict SLR Given TAS'''

def predict_slr(tas_nc_path):
    X_custom = xr.open_dataset(tas_nc_path)

    SLR_custom = pd.DataFrame({'year': np.arange(2015, 2101)})
        
    SLR_custom['5q_dH_dT'] = model_5q.predict(X_custom) 
    SLR_custom['17q_dH_dT'] = model_17q.predict(X_custom) 
    SLR_custom['50q_dH_dT'] = model_50q.predict(X_custom) 
    SLR_custom['83q_dH_dT'] = model_83q.predict(X_custom) 
    SLR_custom['95q_dH_dT'] = model_95q.predict(X_custom) 

    # If want in mm, otherwise remove the * 1000.
    SLR_custom = SLR_custom.set_index('year').cumsum() * 1000 # m to mm.

    return SLR_custom

'''PS SLR'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import LinearRegression 
from glob import glob

data_path = "" # Change to your working directory.

# # Path to save the netcdf file.
# path_output = data_path + 'outputs_ssp245_prediction_pattern2.nc'

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

# TAS in historical, SSP126, SSP370, SSP585.
y_inp_tas = Y["tas"].stack(dims=["lat", "lon"])

# SSP245
test_Y = xr.open_dataset('outputs_ssp245.nc').compute()
test_X = xr.open_dataset('inputs_ssp245.nc').compute()

test_inputs = pd.DataFrame(
    {"CO2": normalize_co2(test_X["CO2"].data),},
    index=test_X["CO2"].coords['time'].data)

ps_tas = LinearRegression(fit_intercept=False)
ps_tas.fit(X['CO2'].to_numpy().reshape(-1, 1), y_inp_tas)

pred_tas = ps_tas.predict(test_X['CO2'].to_numpy().reshape(-1, 1)).reshape(86, 96, 144)

true_tas = test_Y["tas"].mean('member')

ps_tas_global_mean = true_tas.mean(dim=("lat", "lon"))
X_custom = ps_tas_global_mean.values.reshape(-1, 1)

SLR_PS = pd.DataFrame({'year': np.arange(2015, 2101)})

SLR_PS['5q_dH_dT'] = model_5q.predict(X_custom) ### IMPORTANT... if we don't want to go through the trouble 
SLR_PS['17q_dH_dT'] = model_17q.predict(X_custom) ### of training each model, we can save the model weights
SLR_PS['50q_dH_dT'] = model_50q.predict(X_custom) ### (intercept & slope) and just save those in a file and 
SLR_PS['83q_dH_dT'] = model_83q.predict(X_custom) ### do the calculation manually, to make the app more 
SLR_PS['95q_dH_dT'] = model_95q.predict(X_custom) ### more efficient.

SLR_PS = SLR_PS.set_index('year').cumsum() * 1000

'''GP SLR'''
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

X = xr.open_mfdataset([data_path + 'inputs_historical.nc', data_path + 'inputs_ssp585.nc',
                      data_path + 'inputs_ssp126.nc', data_path + 'inputs_ssp370.nc']).compute()

Y = xr.concat([xr.open_dataset(data_path + 'outputs_historical.nc').sel(member=2, method='nearest'),
               xr.open_dataset(data_path + 'outputs_ssp585.nc').sel(member=2,  method='nearest'),
              # xr.open_dataset(data_path + 'outputs_ssp126.nc').sel(member=2,  method='nearest'),
              # xr.open_dataset(data_path + 'outputs_ssp370.nc').sel(member=2,  method='nearest')
              ], dim='time').compute()

# Y =xr.open_mfdataset([data_path + 'outputs_historical.nc', data_path + 'outputs_ssp585.nc',
#                       data_path + 'outputs_ssp126.nc', data_path + 'outputs_ssp370.nc'],
#                       concat_dim='time',
#                      combine='nested'
# ).compute()

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


# Bring the emissions data back together again and normalise
inputs = pd.DataFrame({
    "CO2": normalize_co2(X["CO2"].data),
    "CH4": normalize_ch4(X["CH4"].data)
}, index=X["CO2"].coords['time'].data)

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
bc_solver = Eof(X['BC'])

# Retrieve the leading EOF, expressed as the correlation between the leading
# PC time series and the input SST anomalies at each grid point, and the
# leading PC time series itself.
bc_eofs = bc_solver.eofsAsCorrelation(neofs=5)
bc_pcs = bc_solver.pcs(npcs=5, pcscaling=1)

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
so2_solver = Eof(X['SO2'])

# Retrieve the leading EOF, expressed as the correlation between the leading
# PC time series and the input SST anomalies at each grid point, and the
# leading PC time series itself.
so2_eofs = so2_solver.eofsAsCorrelation(neofs=5)
so2_pcs = so2_solver.pcs(npcs=5, pcscaling=1)

# Convert the Principle Components of the aerosol emissions (calculated above) in to Pandas DataFrames
bc_df = bc_pcs.to_dataframe().unstack('mode')
bc_df.columns = [f"BC_{i}" for i in range(5)]

so2_df = so2_pcs.to_dataframe().unstack('mode')
so2_df.columns = [f"SO2_{i}" for i in range(5)]

# Combine with aerosol EOFs
inputs=pd.concat([inputs, bc_df, so2_df], axis=1)

tas_gp = gp_model(inputs, Y["tas"])
tas_gp.train()

test_Y = xr.open_dataset('outputs_ssp245.nc').compute()
test_X = xr.open_dataset('inputs_ssp245.nc').compute()

test_inputs = pd.DataFrame({
    "CO2": normalize_co2(test_X["CO2"].data),
    "CH4": normalize_ch4(test_X["CH4"].data)
}, index=test_X["CO2"].coords['time'].data)

# Combine with aerosol EOFs
test_inputs=pd.concat([test_inputs, 
                       bc_solver.projectField(test_X["BC"], neofs=5, eofscaling=1).to_dataframe().unstack('mode').rename(columns={i:f"BC_{i}" for i in range(5)}),
                       so2_solver.projectField(test_X["SO2"], neofs=5, eofscaling=1).to_dataframe().unstack('mode').rename(columns={i:f"_{i}" for i in range(5)}),
                       ], axis=1)
# m_tas, _ = tas_gp.predict(test_inputs)

copy = test_inputs.copy()

m_tas, _ = tas_gp.predict(copy)
tas_global_mean = m_tas.mean(dim=("lat", "lon"))  # (86,)

X_custom = tas_global_mean.values.reshape(-1, 1)

SLR_GP = pd.DataFrame({'year': np.arange(2015, 2101)})

SLR_GP['5q_dH_dT'] = model_5q.predict(X_custom) ### IMPORTANT... if we don't want to go through the trouble 
SLR_GP['17q_dH_dT'] = model_17q.predict(X_custom) ### of training each model, we can save the model weights
SLR_GP['50q_dH_dT'] = model_50q.predict(X_custom) ### (intercept & slope) and just save those in a file and 
SLR_GP['83q_dH_dT'] = model_83q.predict(X_custom) ### do the calculation manually, to make the app more 
SLR_GP['95q_dH_dT'] = model_95q.predict(X_custom) ### more efficient.

SLR_GP = SLR_GP.set_index('year').cumsum() * 1000

'''CNN SLR'''

from tensorflow.keras.models import load_model

# Load the saved model in .keras format
cnn_model = load_model("data/cnn_model_default.keras")

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

X_test = xr.open_mfdataset(['inputs_historical.nc',
                            'inputs_ssp245.nc']).compute()

# Normalize data 
for var in ['CO2', 'CH4', 'SO2', 'BC']: 
    var_dims = X_test[var].dims
    X_test = X_test.assign({var: (var_dims, normalize(X_test[var].data, var, meanstd_inputs))}) 
    
X_test_np = input_for_training(X_test, skip_historical=False, len_historical=len_historical)  

# Make predictions using trained model 
m_pred = cnn_model.predict(X_test_np)
# reshape to xarray 
m_pred = m_pred.reshape(m_pred.shape[0], m_pred.shape[2], m_pred.shape[3])
m_pred = xr.DataArray(m_pred, dims=['time', 'lat', 'lon'], coords=[X_test.time.data[slider-1:], X_test.latitude.data, X_test.longitude.data])
m_pred = m_pred.transpose('lat', 'lon', 'time').sel(time=slice(2015, 2101)).to_dataset(name="tas")
cnn = m_pred.mean(dim=("lat", "lon"))
X_custom = cnn["tas"].values.reshape(-1, 1)

SLR_cnn = pd.DataFrame({'year': np.arange(2015, 2101)})

SLR_cnn['5q_dH_dT'] = model_5q.predict(X_custom) ### IMPORTANT... if we don't want to go through the trouble 
SLR_cnn['17q_dH_dT'] = model_17q.predict(X_custom) ### of training each model, we can save the model weights
SLR_cnn['50q_dH_dT'] = model_50q.predict(X_custom) ### (intercept & slope) and just save those in a file and 
SLR_cnn['83q_dH_dT'] = model_83q.predict(X_custom) ### do the calculation manually, to make the app more 
SLR_cnn['95q_dH_dT'] = model_95q.predict(X_custom) ### more efficient.

SLR_cnn = SLR_cnn.set_index('year').cumsum() * 1000

'''RF SLR'''
from esem import rf_model
train_files = [ "historical", "ssp585", "ssp126", "ssp370", ]

def create_predictor_data(data_sets, n_eofs=5):
    """
    Args:
        data_sets list(str): names of datasets
        n_eofs (int): number of eofs to create for aerosol variables
    """
    # Create training and testing arrays
    if isinstance(data_sets, str):
        data_sets = [data_sets]
    X = xr.concat([xr.open_dataset(data_path + f"inputs_{file}.nc") for file in data_sets], dim='time')
    X = X.assign_coords(time=np.arange(len(X.time)))

    # Compute EOFs for BC
    bc_solver = Eof(X['BC'])
    bc_eofs = bc_solver.eofsAsCorrelation(neofs=n_eofs)
    bc_pcs = bc_solver.pcs(npcs=n_eofs, pcscaling=1)

    # Compute EOFs for SO2
    so2_solver = Eof(X['SO2'])
    so2_eofs = so2_solver.eofsAsCorrelation(neofs=n_eofs)
    so2_pcs = so2_solver.pcs(npcs=n_eofs, pcscaling=1)

    # Convert to pandas
    bc_df = bc_pcs.to_dataframe().unstack('mode')
    bc_df.columns = [f"BC_{i}" for i in range(n_eofs)]

    so2_df = so2_pcs.to_dataframe().unstack('mode')
    so2_df.columns = [f"SO2_{i}" for i in range(n_eofs)]

    # Bring the emissions data back together again and normalise
    inputs = pd.DataFrame({
        "CO2": normalize_co2(X["CO2"].data),
        "CH4": normalize_ch4(X["CH4"].data)
    }, index=X["CO2"].coords['time'].data)

    # Combine with aerosol EOFs
    inputs = pd.concat([inputs, bc_df, so2_df], axis=1)
    return inputs, (so2_solver, bc_solver)

def create_predictdand_data(data_sets):
    if isinstance(data_sets, str):
        data_sets = [data_sets]
    Y = xr.concat([xr.open_dataset(f"outputs_{file}.nc") for file in data_sets], dim='time').mean("member")
    # Convert the precip values to mm/day
    Y["pr"] *= 86400
    Y["pr90"] *= 86400
    return Y

# Create training and testing arrays
X, solvers = create_predictor_data(train_files)
Y = create_predictdand_data(train_files)

rf_tas = rf_model(X, Y['tas'], random_state=0, bootstrap=True, max_features='sqrt',
                           **{'n_estimators': 250, 'min_samples_split': 5, 'min_samples_leaf': 7,  'max_depth': 5,})

rf_tas.train()

def get_test_data(file, eof_solvers, n_eofs=5):
    """
    Args:
        file str: name of datasets
        n_eofs (int): number of eofs to create for aerosol variables
        eof_solvers (Eof_so2, Eof_bc): Fitted Eof objects to use for projection
    """
        
    # Create training and testing arrays
    X = xr.open_dataset(f"inputs_{file}.nc")
        
    so2_pcs = eof_solvers[0].projectField(X["SO2"], neofs=5, eofscaling=1)
    so2_df = so2_pcs.to_dataframe().unstack('mode')
    so2_df.columns = [f"SO2_{i}" for i in range(n_eofs)]

    bc_pcs = eof_solvers[1].projectField(X["BC"], neofs=5, eofscaling=1)
    bc_df = bc_pcs.to_dataframe().unstack('mode')
    bc_df.columns = [f"BC_{i}" for i in range(n_eofs)]

    # Bring the emissions data back together again and normalise
    inputs = pd.DataFrame({
        "CO2": normalize_co2(X["CO2"].data),
        "CH4": normalize_ch4(X["CH4"].data)
    }, index=X["CO2"].coords['time'].data)

    # Combine with aerosol EOFs
    inputs = pd.concat([inputs, bc_df, so2_df], axis=1)
    return inputs
    
X_test = get_test_data('ssp245', solvers)

m_out_tas, _ = rf_tas.predict(X_test)

rf_global_mean = m_out_tas.mean(dim=("lat", "lon"))  # (86,)

X_custom = rf_global_mean.values.reshape(-1, 1)

SLR_RF = pd.DataFrame({'year': np.arange(2015, 2101)})

SLR_RF['5q_dH_dT'] = model_5q.predict(X_custom) ### IMPORTANT... if we don't want to go through the trouble 
SLR_RF['17q_dH_dT'] = model_17q.predict(X_custom) ### of training each model, we can save the model weights
SLR_RF['50q_dH_dT'] = model_50q.predict(X_custom) ### (intercept & slope) and just save those in a file and 
SLR_RF['83q_dH_dT'] = model_83q.predict(X_custom) ### do the calculation manually, to make the app more 
SLR_RF['95q_dH_dT'] = model_95q.predict(X_custom) ### more efficient.

SLR_RF = SLR_RF.set_index('year').cumsum() * 1000

'''Figures'''

temp = pd.DataFrame({"year": range(2015, 2101), 
                       "cnn pred": SLR_cnn['50q_dH_dT'].to_numpy(),
                   "linear pred": SLR_PS['50q_dH_dT'].to_numpy(),
                    "GP pred": SLR_GP['50q_dH_dT'].to_numpy(),
                    "RF pred": SLR_RF['50q_dH_dT'].to_numpy()
                    }).set_index("year")

plt.plot(val_50q['expected_SL'], label='Expected', linestyle=':', color='black', linewidth=2)

plt.plot(temp['linear pred'], label='PS', linestyle='--', color='orange', alpha=0.8)
plt.plot(temp['GP pred'], label='GP', linestyle='--', color='royalblue', alpha=0.8)
plt.plot(temp['RF pred'], label='RF', linestyle='--', color='green', alpha=0.8)
plt.plot(temp['cnn pred'], label='CNN', linestyle='--', color='hotpink', alpha=0.8)

plt.xlabel('Year')
plt.ylabel('Sea Level Rise (mm)')
plt.title('Projected Sea Level Rise for SSP 245 Using Emulators')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("ssp245_emulator_preds.png", dpi=600, bbox_inches='tight')
plt.show()

'''Keeping Greenhouse Gases Constant'''
cnn_df = pd.read_csv("data/CNN_Carbon_4520_Preds.csv").set_index('year')
gp_df = pd.read_csv("data/GP_Carbon_4520_Preds.csv").set_index('year')
rf_df = pd.read_csv("data/RF_Carbon_4520_Preds.csv").set_index('year')
ps_df = pd.read_csv("data/PS_Carbon_4520_Preds.csv").set_index('year')

plt.plot(val_50q['expected_SL'], label='Expected', linestyle=':', color='black', linewidth=2)

plt.plot(ps_df['50q_dH_dT'], label='PS', linestyle='--', color='orange', alpha=0.8)
plt.plot(gp_df['50q_dH_dT'], label='GP', linestyle='--', color='royalblue', alpha=0.8)
plt.plot(rf_df['50q_dH_dT'], label='RF', linestyle='--', color='green', alpha=0.8)
plt.plot(cnn_df['50q_dH_dT'], label='CNN', linestyle='--', color='hotpink', alpha=0.8)

plt.xlabel('Year')
plt.ylabel('Sea Level Rise (mm)')
plt.title('Projected Sea Level Rise Keeping Greenhouse Gases Fixed')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("2025_fixed_emulator_preds.png", dpi=600, bbox_inches='tight')
plt.show()
