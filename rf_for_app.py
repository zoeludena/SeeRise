import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof
import esem

import warnings
warnings.filterwarnings('ignore')
from esem import rf_model
from esem.data_processors import Whiten, Normalise

from SLR import model_5q, model_17q, model_50q, model_83q, model_95q

data_path_1 = "data/inputs_outputs/"

X = xr.open_mfdataset([data_path_1 + 'inputs_historical.nc', data_path_1 + 'inputs_ssp585.nc',
                      data_path_1 + 'inputs_ssp126.nc', data_path_1 + 'inputs_ssp370.nc']).compute()

Y = xr.concat([xr.open_dataset(data_path_1 + 'outputs_historical.nc').sel(member=2, method='nearest'),
               xr.open_dataset(data_path_1 + 'outputs_ssp585.nc').sel(member=2,  method='nearest'),
              # xr.open_dataset(data_path_1 + 'outputs_ssp126.nc').sel(member=2,  method='nearest'),
              # xr.open_dataset(data_path_1 + 'outputs_ssp370.nc').sel(member=2,  method='nearest')
              ], dim='time').compute()

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

print("Start making BC and SO2 Solvers")
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

tas_rf = rf_model(inputs, Y['tas'], 
                  random_state=0, 
                  bootstrap=True, 
                  max_features=0.6, 
                  **{'n_estimators': 1200, 
                     'min_samples_split': 5, 
                     'min_samples_leaf': 4,  
                     'max_depth': 12,})

print("Created random forest model")

tas_rf.train()

print("Train model")

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

possible_carbons = np.arange(0, 9510, 10)
possible_carbons = np.array([4520])

def create_rf_carbon_preds(possible_carbons):

    copy = {'CO2': np.full(86, test_inputs.loc[2025].values[0]),
       'CH4': np.full(86, test_inputs.loc[2025].values[1]),
       ('pseudo_pcs', 'BC_0'):np.full(86, test_inputs.loc[2025].values[2]),
       ('pseudo_pcs', 'BC_1'):np.full(86, test_inputs.loc[2025].values[3]),
       ('pseudo_pcs', 'BC_2'):np.full(86, test_inputs.loc[2025].values[4]),
        ('pseudo_pcs', 'BC_3'):np.full(86, test_inputs.loc[2025].values[5]),
        ('pseudo_pcs', 'BC_4'):np.full(86, test_inputs.loc[2025].values[6]),
        ('pseudo_pcs', '_0'):np.full(86, test_inputs.loc[2025].values[7]),
        ('pseudo_pcs', '_1'):np.full(86, test_inputs.loc[2025].values[8]),
        ('pseudo_pcs', '_2'):np.full(86, test_inputs.loc[2025].values[9]),
        ('pseudo_pcs', '_3'):np.full(86, test_inputs.loc[2025].values[10]),
        ('pseudo_pcs', '_4'):np.full(86, test_inputs.loc[2025].values[11])
    }

    copy = pd.DataFrame(copy)
    
    for carbon in possible_carbons:
        last_hist_CO2 = xr.open_dataset(data_path_1 + 'inputs_historical.nc')['CO2'].data[-1]
        co2 = np.linspace(last_hist_CO2, carbon, 86)
        # MUST NORMALIZE CO2    
        copy["CO2"] = np.full(copy.shape[0], normalize_co2(co2)) # use the estimated trajectory
        m_tas, _ = tas_rf.predict(copy)
        tas_global_mean = m_tas.mean(dim=("lat", "lon"))  # (86,)
        X_custom = tas_global_mean.values.reshape(-1, 1)
        
        SLR_custom = pd.DataFrame({'year': np.arange(2015, 2101)})
        
        SLR_custom['5q_dH_dT'] = model_5q.predict(X_custom) ### IMPORTANT... if we don't want to go through the trouble 
        SLR_custom['17q_dH_dT'] = model_17q.predict(X_custom) ### of training each model, we can save the model weights
        SLR_custom['50q_dH_dT'] = model_50q.predict(X_custom) ### (intercept & slope) and just save those in a file and 
        SLR_custom['83q_dH_dT'] = model_83q.predict(X_custom) ### do the calculation manually, to make the app more 
        SLR_custom['95q_dH_dT'] = model_95q.predict(X_custom) ### more efficient.
        
        SLR_custom = SLR_custom.set_index('year').cumsum() * 1000 #<- if want in mm, otherwise remove.

        # Uncomment line below and create appropriate GP_245_Linear folder to save multiple CSVs
        # SLR_custom.to_csv(f"data/RF_245_linear/RF_Carbon_{carbon}_Preds.csv")

        if carbon == 4520:
            SLR_custom.to_csv(f"data/RF_Carbon_{carbon}_Preds.csv")
            print("RF model for sea level rise using SSP 245 with 4520 gigatons of cumulative carbon dioxide csv has been created!")

create_rf_carbon_preds(possible_carbons)
