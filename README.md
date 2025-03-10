# SeeRise: Visualizing Emulated Sea Level Rise on Coastal Regions

## Dependencies

- Users are encouraged to use NCAR Casper Login*.
- Data is available from the sixth Coupled Model Intercomparison Product (CMIP6). It is publicly archived and available, but the data is readily available on Casper. Here is a link to the data on the [Earth System Grid Federation Portal at Centre for Environmental Data Analysis](https://esgf-ui.ceda.ac.uk/cog/projects/esgf-ceda/) and the [cloud](https://registry.opendata.aws/cmip6/).
- Run `pip install -r requirements.txt` to download any missing Python dependencies.

(**NCAR Casper Login Aside**: You can create an account by following the directions on the [Casper website](https://arc.ucar.edu/docs).)

## Emulator Replication

**Data Access**

The processed training, validation and test data can be obtained from [Zenodo](https://doi.org/10.5281/zenodo.5196512).

- Download `test.tar.gz` and `train_val.tar.gz`.
- Decompressing the two files.
- Upload all `.nc` files in `train_val` and `test` onto CASPER and place them in the same directory.
    - You can also work locally, put the files from `.nc` files into this repository `data/inputs_outputs`.

The historical data can also be obtained from [Zenodo](https://zenodo.org/records/3862995). This one is a data supplement for "The causes of sea-level rise since 1900."
- Download `global_basin_timeseries.xlsx` and upload it in the same directory as your notebooks and `.nc` files.

The NASA data can be obtained using the [Sea Level Projection Tool](https://sealevel.nasa.gov/ipcc-ar6-sea-level-projection-tool?type=global). Beneath each visualization you can download the data as an Excel file.
- Download `ipcc_ar6_sea_level_projection_global.xlsx` and upload it in the same directory as your notebooks and `.nc` files.

**Models**
- Download [`utils.py`](Emulators/utils.py) and upload onto Casper.
- Pattern Scaling
  - Download [`pattern_scaling_model.py`](Emulators/pattern_scaling_model.py) and upload onto Casper.
  - Update `data_path` location to directory of `.nc` files.
  - Run the  `py` file to see the linear model and outputs.
- Gaussian Process
  - Download [`simple_GP_model.py`](Emulators/simple_GP_model.py) and upload onto Casper.
  - Update `data_path` location to directory of `.nc` files.
  - Run the  `py` file to see the gaussian process model and outputs.
- Random Forest Model
  - Download [`RF_model_ESEm.py`](Emulators/RF_model_ESEm.py) and upload onto Casper.
  - Place `utils.py` and the  `py` file in the same directory as the `.nc` files.
  - Run the notebook to see the random forest model and outputs.
- CNN-LTSM
  - Download [`CNN-LTSM_model.py`](Emulators/CNN-LTSM_model.py) and upload onto Casper.
  - Place the `py` file in same directory as the `.nc` files.
  - Run the notebook to see the neural network’s model and outputs.

## Sea Level Rise Replication

The sea level rise model can be run locally, with appropriate dependencies installed. Follow instructions in `data/inputs_outputs/instructions.md` to prepare the data necessary for the model. Run `SLR.py` in the root directory to reproduce our sea level rise model that uses Rahmstorf’s approach.

To see the entire sea level rise prediction pipeline and get actual predictions from our model (which are the same as predictions on our interactive streamlit app), run the corresponding python file for each climate emulator:
- `ps_for_app.py`
- `gp_for_app.py`
- `rf_for_app.py`
- `cnn_for_app.py`

Uncomment the line toward the end of each file and under `“# Uncomment line below and create appropriate X_245_Linear folder to save multiple CSVs”` to generate and store CSV files of sea level rise predictions shown in our app.
