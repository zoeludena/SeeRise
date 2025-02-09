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

Ensure the [`Sea_Level_Rise.py`](Sea_Level_Rise.py) is in the same directors as the `.nc` files, `global_basin_timeseries.xlsx`, and `ipcc_ar6_sea_level_projection_global.xlsx`.

This `py` file contains code that will take emulator produced temperature (`tas`) to year 2100 and apply “A Semi-Empirical Approach to Projecting Future Sea-Level Rise” by Stefan Rahmstorf. From there we compare our emulator’s prediction for sea level rise with NASA’s Sea Level Projection.

