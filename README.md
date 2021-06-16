# Two-State GUI
A GUI for fitting two-state hidden Markov models on one-dimensional time series.

This user interface allows to:
- Visualize data and a fitted HMM
- Manually correct the fitted state of time intervals by simply clicking on the corresponding curve.
- Fit a HMM on the data.
- Export the results and summary statistics as CSV files.


## Installation
From a dedicated virtual environment, clone the package and install the requirements:
```bash
$ git clone https://github.com/rfayat/GUI_outlier_correction.git
$ cd GUI_outlier_correction
$ pip install -r requirements.txt
# Package for HMM fitting
$ pip install git+git://github.com/lindermanlab/ssm.git
```

## Running the app
From the cloned repository folder, run:
```bash
$ python -m two_state_gui.app
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
```
The app is now running and can be accessed by visiting the displayed address (http://127.0.0.1:8050/).

**Warning** The computer serving the app needs to be connected to the internet for loading the templates and graphical components.

## Bug report
If you find a bug, please [open an issue](https://github.com/rfayat/two_state_gui/issues) on the github repository and provide as much details as possible to replicate it (files used, outputs in the console or the dash application...).
