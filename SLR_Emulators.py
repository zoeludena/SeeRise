'''Import Statements'''

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
from glob import glob

from sklearn.linear_model import LinearRegression
from eofs.xarray import Eof
import esem
from esem import gp_model, rf_model
from esem.data_processors import Whiten, Normalise
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')
from SLR import model_5q, model_17q, model_50q, model_83q, model_95q, val_50q

# Change this path to where your emulator inputs and outputs are stored. We are training
# the SLR model on historical data and predicting on SSP data.
data_path_1 = "data/inputs_outputs/"
# Change this data path to where you will keep the global_basin_timeseries and ipcc_ar6_sea_level_projection_global excel files
data_path_2 = "data/"
# Change this datapath to where you keep the quantile csv files
data_path_3 = "SLR_Training/"
# Replace with the datapath that holds the neural network (from where you saved it after running the emulator)
data_path_4 = "data/"
# Change this path to where the CSV files are from emulator_for_app.py files
data_path_5 = "data/"
# Change this path to where you want the images from figures saved
data_path_6 = "images/"

'''PS SLR'''

# Path to testing data (ssp245).
pathssp245_Y = data_path_1 + 'outputs_ssp245.nc'
pathssp245_X = data_path_1 + 'inputs_ssp245.nc'

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

inputs = glob(data_path_1 + "inputs_s*.nc")
SECONDS_IN_YEAR = 60*60*24*365 #s

# Get one combined historical + ssp585 + ssp126 + ssp370 timeseries for now.
X = xr.concat([
    xr.open_dataset(data_path_1 + 'inputs_historical.nc'), 
    xr.open_dataset(data_path_1 + 'inputs_ssp585.nc'),
    xr.open_dataset(data_path_1 + 'inputs_ssp126.nc'),
    xr.open_dataset(data_path_1 + 'inputs_ssp370.nc')], dim='time').compute()

# Take the 2nd ensemble member for the historical (the first one has some 
# missing DTR values for some reason...) and the 1st one for ssp585, ssp126, 
# ssp370.
Y = xr.concat([
    xr.open_dataset(data_path_1 + 'outputs_historical.nc').sel(member=2), 
    xr.open_dataset(data_path_1 + 'outputs_ssp585.nc').sel(member=1),
    xr.open_dataset(data_path_1 + 'outputs_ssp126.nc').sel(member=1),
    xr.open_dataset(data_path_1 + 'outputs_ssp370.nc').sel(member=1)], 
    dim='time').compute()

# TAS in historical, SSP126, SSP370, SSP585.
y_inp_tas = Y["tas"].stack(dims=["lat", "lon"])

# SSP245
test_Y = xr.open_dataset(data_path_1 + 'outputs_ssp245.nc').compute()
test_X = xr.open_dataset(data_path_1 + 'inputs_ssp245.nc').compute()

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

print("PS Model Finished Predicting Sea Level Rise")

'''GP SLR'''

X = xr.open_mfdataset([data_path_1 + 'inputs_historical.nc', data_path_1 + 'inputs_ssp585.nc',
                      data_path_1 + 'inputs_ssp126.nc', data_path_1 + 'inputs_ssp370.nc']).compute()

# Shape issue when attempt to use all other outputs than SSP 245
Y = xr.concat([xr.open_dataset(data_path_1 + 'outputs_historical.nc').sel(member=2, method='nearest'),
               xr.open_dataset(data_path_1 + 'outputs_ssp585.nc').sel(member=2,  method='nearest'),
              ], dim='time').compute()

min_co2 = 0.
max_co2 = 9500

min_ch4 = 0.
max_ch4 = 0.8

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

test_Y = xr.open_dataset(data_path_1 + 'outputs_ssp245.nc').compute()
test_X = xr.open_dataset(data_path_1 + 'inputs_ssp245.nc').compute()

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

print("GP Model Finished Predicting Sea Level Rise")

'''CNN SLR'''

# Load the saved model in .keras format
cnn_model = load_model(data_path_4 + "cnn_model_default.keras")

simus = ['ssp126',
         'ssp370',
         'ssp585',
         'hist-GHG',
         'hist-aer']

slider = 10

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

    # print(input_xr.dims, simu)

    # Append to list 
    X_train.append(input_xr)
    Y_train.append(output_xr)

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

X_test = xr.open_mfdataset([data_path_1 + 'inputs_historical.nc',
                            data_path_1 + 'inputs_ssp245.nc']).compute()

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

print("For meanstd_inputs:")
for var in ['CO2', 'CH4', 'SO2', 'BC']:
    # To not take the historical data into account several time we have to slice the scenario datasets
    # and only keep the historical data once (in the first ssp index 0 in the simus list)
    array = np.concatenate([X_train[i][var].data for i in [0, 3, 4]] + 
                           [X_train[i][var].sel(time=slice(len_historical, None)).data for i in range(1, 3)])
    print(f"{var} mean: {array.mean()}")
    print(f"{var} std {array.std()}")
    meanstd_inputs[var] = (array.mean(), array.std())

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

print("CNN Model Finished Predicting Sea Level Rise")

'''RF SLR'''

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
    X = xr.concat([xr.open_dataset(data_path_1 + f"inputs_{file}.nc") for file in data_sets], dim='time')
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
    Y = xr.concat([xr.open_dataset(data_path_1 + f"outputs_{file}.nc") for file in data_sets], dim='time').mean("member")
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
    X = xr.open_dataset(data_path_1 + f"inputs_{file}.nc")
        
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

print("RF Model Finished Predicting Sea Level Rise")

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

plt.savefig(data_path_6 + "ssp245_emulator_preds.png", dpi=600, bbox_inches='tight')
print("Compare Emulators to Expected Image Done")

'''Keeping Greenhouse Gases Constant'''
cnn_df = pd.read_csv(data_path_5 + "CNN_Carbon_4520_Preds.csv").set_index('year')
gp_df = pd.read_csv(data_path_5 + "GP_Carbon_4520_Preds.csv").set_index('year')
rf_df = pd.read_csv(data_path_5 + "RF_Carbon_4520_Preds.csv").set_index('year')
ps_df = pd.read_csv(data_path_5 + "PS_Carbon_4520_Preds.csv").set_index('year')

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

plt.savefig(data_path_6 + "2025_fixed_emulator_preds.png", dpi=600, bbox_inches='tight')

print("Greenhouse Gases Fixed Image Done")