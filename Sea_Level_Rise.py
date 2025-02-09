'''Import Statements'''

import datetime as dt 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge

from eofs.xarray import Eof
from glob import glob

'''Process Emulator Inputs'''

# Change this path to where your emulator outputs are stored. We are training
# the SLR model on historical data and predicting on SSP data.
data_path = ""

inputs = glob(data_path + "inputs_s*.nc")
SECONDS_IN_YEAR = 60*60*24*365 #s

fig, axes = plt.subplots(2, 2, figsize=(12,12))

for input in inputs:
    label=input.split('_')[1][:-3]
    X = xr.open_dataset(input)
    x = range(2015, 2101)

    # Need this block and the next for the weights.
    weights = np.cos(np.deg2rad(X.latitude))
    
    axes[0, 0].plot(x, X['CO2'].data, label=label)
    axes[0, 0].set_ylabel("Cumulative anthropogenic CO2 \nemissions since 1850 (GtCO2)")
    axes[0, 1].plot(x, X['CH4'].data, label=label)
    axes[0, 1].set_ylabel("Anthropogenic CH4 \nemissions (GtCH4 / year)")
    # FIXME: Not sure where this factor of 1000 comes from...! Maybe the CEDS data is really g/m-2/s?
    axes[1, 0].plot(x, X['SO2'].weighted(weights).sum(['latitude', 'longitude']).data*SECONDS_IN_YEAR*1e-9, label=label)
    axes[1, 0].set_ylabel("Anthropogenic SO2 \nemissions (GtSO2 / year)")
    axes[1, 1].plot(x, X['BC'].weighted(weights).sum(['latitude', 'longitude']).data*SECONDS_IN_YEAR*1e-9, label=label)
    axes[1, 1].set_ylabel("Anthropogenic BC \nemissions (GtBC / year)")

axes[0, 0].set_title('CO2')
axes[0, 1].set_title('CH4')
axes[1, 0].set_title('SO2')
axes[1, 1].set_title('BC')
axes[0, 0].legend()
plt.tight_layout()

'''More Input Processing'''

# Get one combined historical + ssp585 + ssp126 + ssp370 timeseries for now
X = xr.concat([xr.open_dataset(data_path + 'inputs_historical.nc'), xr.open_dataset(data_path + 'inputs_ssp585.nc'),xr.open_dataset(data_path+ 'inputs_ssp126.nc'),xr.open_dataset(data_path+ 'inputs_ssp370.nc')], dim='time').compute()

# Take the 2nd ensemble member for the historical (the first one has some missing DTR values for some reason...) and the 1st one for ssp585, ssp126, ssp370
Y = xr.concat([xr.open_dataset(data_path + 'outputs_historical.nc').sel(member=2), xr.open_dataset(data_path + 'outputs_ssp585.nc').sel(member=1),xr.open_dataset(data_path+ 'outputs_ssp126.nc').sel(member=1),xr.open_dataset(data_path+ 'outputs_ssp370.nc').sel(member=1)], dim='time').compute()

# Convert the precip values to mm/day
Y["pr"] *= 86400
Y["pr90"] *= 86400

X["time"]=np.arange(1,424) 
Y["time"]=np.arange(1,424)

# Need the weights here as well. Very important in order to weight
# the output file TAS variables.
weights = np.cos(np.deg2rad(Y.lat))
global_mean_temp = Y['tas'].weighted(weights).mean(['lat', 'lon']).to_pandas()

historical_tas = xr.open_dataset("outputs_historical.nc")["tas"].weighted(weights).mean(['lat', 'lon', 'member']).to_pandas()
ssp126_tas = xr.open_dataset("outputs_ssp126.nc")["tas"].weighted(weights).mean(['lat', 'lon', 'member']).to_pandas()
ssp245_tas = xr.open_dataset("outputs_ssp245.nc")["tas"].weighted(weights).mean(['lat', 'lon', 'member']).to_pandas()
ssp370_tas = xr.open_dataset("outputs_ssp370.nc")["tas"].weighted(weights).mean(['lat', 'lon', 'member']).to_pandas()
ssp585_tas = xr.open_dataset("outputs_ssp585.nc")["tas"].weighted(weights).mean(['lat', 'lon', 'member']).to_pandas()

'''Plot of Historical Sea Level'''

more_sl = pd.read_excel(data_path + "global_basin_timeseries.xlsx", index_col=0)
gmsl = more_sl[["Observed GMSL [mean]"]]
gmsl_anom = gmsl - gmsl.loc[1900]

plt.plot(gmsl_anom, label='Observed GMSL', color='dodgerblue', linestyle='-')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Sea Level Rise (mm)')
plt.title('Historical Sea Level Rise')
plt.show()

'''Regress SLR on Temperature Anomaly'''

# Features (temperature anomaly) and target (rate of sea level rise)
X_hist = historical_tas.loc[1901:2014].values.reshape(-1, 1)
y = gmsl_anom.diff().loc[1901:2014].values.reshape(-1, 1)

# Fit the linear regression model
hist_model = LinearRegression()
hist_model.fit(X_hist, y)

# Extract the parameter 'a'
a = hist_model.coef_[0]
intercept = hist_model.intercept_

hist_sl = pd.DataFrame({"year": range(1901, 2015), "pred dH_dt (mm/yr)": hist_model.predict(X_hist).flatten()})
hist_sl = hist_sl.set_index("year")
hist_sl = hist_sl.join(gmsl_anom, how='left')
hist_sl['pred_sea_level_rise'] = intercept + hist_sl['pred dH_dt (mm/yr)'].cumsum()

MSE_TAS = sum((hist_sl['Observed GMSL [mean]'] - hist_sl['pred_sea_level_rise']) ** 2) / hist_sl.shape[0]

plt.plot(hist_sl['Observed GMSL [mean]'], 
         label='Observed', linestyle='-', color='dodgerblue')
plt.plot(hist_sl['pred_sea_level_rise'], 
         label='Predicted', linestyle='--', color='black')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Sea Level Rise (mm)')
plt.title('Predicted and Historical Sea Level Rise')
plt.show()
print(MSE_TAS ** 0.5)

'''Using Emulators'''

'''CNN - From the Climate Bench'''
default_tas_cnn = xr.open_dataset("default_outputs_ssp245_predict_tas.nc").weighted(weights).mean(['lat', 'lon']).to_pandas()['tas']

'''Gaussian Process Emulator'''
tas_gp = gp_model(X_hist, y)
tas_gp.train()

X = xr.open_mfdataset([data_path + 'inputs_historical.nc', data_path + 'inputs_ssp585.nc',
                      data_path + 'inputs_ssp126.nc', data_path + 'inputs_ssp370.nc']).compute()

Y = xr.concat([xr.open_dataset(data_path + 'outputs_historical.nc').sel(member=2),
               xr.open_dataset(data_path + 'outputs_ssp585.nc').sel(member=1),
              # xr.open_dataset(data_path + 'outputs_ssp126.nc').sel(member=1),
              # xr.open_dataset(data_path + 'outputs_ssp370.nc').sel(member=1)
              ], dim='time').compute()

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

# Bring the emissions data back together again and normalise
leading_historical_inputs = pd.DataFrame({
    "CO2": normalize_co2(X["CO2"].data),
    "CH4": normalize_ch4(X["CH4"].data)
}, index=X["CO2"].coords['time'].data)

# Combine with aerosol EOFs
leading_historical_inputs=pd.concat([leading_historical_inputs, bc_df, so2_df], axis=1)

from esem import gp_model
from esem.data_processors import Whiten, Normalise

tas_gp = gp_model(leading_historical_inputs, Y["tas"])
tas_gp.train()

# Get the test data (NOT visible to contestants)

#Should this be the test data? Like data in the test file?
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

m_tas, _ = tas_gp.predict(test_inputs)

tas_global_mean = m_tas.mean(dim=("lat", "lon"))  # (86,)

'''Random Forest Emulator'''

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

rf_tas_global_mean = m_out_tas.mean(dim=("lat", "lon"))  # (86,)

emulators = pd.DataFrame({"year": range(2015, 2101), 
                       "default pred cnn": hist_model.predict(default_tas_cnn.values.reshape(-1, 1)).flatten(),
                   "linear pred": hist_model.predict(ssp245_tas.values.reshape(-1, 1)).flatten(),
                    "GP pred": hist_model.predict(tas_global_mean.values.reshape(-1, 1)).flatten(),
                    "RF pred": hist_model.predict(rf_tas_global_mean.values.reshape(-1, 1)).flatten()}).set_index("year").cumsum()

plt.plot(emulators['default pred cnn'], 
         label='CNN SSP 245', linestyle='--', color='grey')
plt.plot(emulators['linear pred'], 
         label='Linear SSP 245', linestyle='--', color='royalblue')
plt.plot(emulators['GP pred'], 
         label='GP SSP 245', linestyle='--', color='orange')
plt.plot(emulators['RF pred'], 
         label='RF SSP 245', linestyle='--', color='lightcoral')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Sea Level Rise (mm)')
plt.title('Projected Sea Level Rise for SSP 245 Using Emulators')
plt.show()

'''Compare Emulators to NASA'''
# Load data
nasa_slc = (pd.read_excel("ipcc_ar6_sea_level_projection_global.xlsx", sheet_name='Total')
          .drop(columns=['process', 2110, 2120, 2130, 2140, 2150])
          )
nasa_slc = (nasa_slc[(nasa_slc['quantile'].isin([5, 50, 95])) & (nasa_slc['scenario'] == 'ssp245') & (nasa_slc['confidence'] != 'low')]
           .drop(columns=['confidence']))

# Define quantiles
quantiles = [5, 50, 95]

# Identify available years
available_years = [col for col in nasa_slc.columns if isinstance(col, int)]

# Extract sea level rise data for SSP 245
nasa_slr_dict = {q: np.nan * np.ones(len(available_years)) for q in quantiles}
for q in quantiles:
    values = nasa_slc[(nasa_slc["quantile"] == q)][available_years].values
    if values.size > 0:
        nasa_slr_dict[q] = values.flatten()

# Create a DataFrame for plots
years = np.arange(2020, 2101, 10)
nasa_slr = pd.DataFrame({
    "year": years,
    "ssp245_median": (nasa_slr_dict[50]*1000),
    "ssp245_low": (nasa_slr_dict[5]*1000),
    "ssp245_high": (nasa_slr_dict[95]*1000),
})

# Load emulator predictions (assuming 'temp' DataFrame is available)
# Columns: ['default pred cnn', 'linear pred', 'GP pred', 'RF pred']

# Define color for SSP 245
ssp245_color = "rgba(128, 128, 128, 0.1)"

# Create figure
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=nasa_slr["year"].tolist() + nasa_slr["year"].tolist()[::-1],
    y=nasa_slr["ssp245_high"].tolist() + nasa_slr["ssp245_low"].tolist()[::-1],
    fill="toself",
    fillcolor=ssp245_color,
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip",
    name="SSP 245 Confidence Interval",
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=nasa_slr["year"],
    y=nasa_slr["ssp245_median"],
    mode="lines",
    line=dict(color=ssp245_color.replace(", 0.1)", ", 1)"), width=2),
    name="NASA",
    hoverinfo='y+name'
))

# Add Emulator Predictions
fig.add_trace(go.Scatter(
    x=temp.index,
    y=temp['default pred cnn'],
    mode='lines',
    name='CNN',
    line=dict(color='purple', dash='dash'),
    hoverinfo='y+name'
))
fig.add_trace(go.Scatter(
    x=temp.index,
    y=temp['linear pred'],
    mode='lines',
    name='Linear',
    line=dict(color='royalblue', dash='dash'),
    hoverinfo='y+name'
))
fig.add_trace(go.Scatter(
    x=temp.index,
    y=temp['GP pred'],
    mode='lines',
    name='GP',
    line=dict(color='orange', dash='dash'),
    hoverinfo='y+name'
))
fig.add_trace(go.Scatter(
    x=temp.index,
    y=temp['RF pred'],
    mode='lines',
    name='RF',
    line=dict(color='lightcoral', dash='dash'),
    hoverinfo='y+name'
))

# Update layout
fig.update_layout(
    title="Projected Sea Level Rise Comparison",
    xaxis_title="Year",
    yaxis_title="Sea Level Rise (mm)",
    template="seaborn",
    legend=dict(
        title=dict(text="SSP 245 Predictions", font=dict(size=14)),
        font=dict(size=13),
        x=1.02,
        y=1,
        xanchor="left",
        yanchor="top"
    ),
    showlegend=True,
    hovermode="x unified"  # Show all values when hovering over a single year
)

fig.write_html("ssp245_projection.html", full_html=True, include_plotlyjs="cdn", auto_open=False, config={"responsive": True})