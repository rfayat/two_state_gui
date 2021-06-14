"""Tools for handling file inputs and outputs.

Author: Romain Fayat, June 2021
"""
import pandas as pd
import io
from .data_handling import HMM_State_Handler
from functools import wraps


def add_success_output(f):
    "Add a boolean indicating whether the function's execution succeeded."
    @wraps(f)
    def g(*args, **kwargs):
        try:
            return f(*args, **kwargs), True
        except Exception as e:
            print(e)
            return None, False
    return g


def parse_content(contents, filename):
    """Parse contents obtained using a dash upload component.

    Source
    ------
    https://dash.plotly.com/dash-core-components/upload
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'xls' in filename:
        # Assume that the user uploaded an excel file
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        # Assume that the user uploaded a CSV file
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8'))
        )
    return df


@add_success_output
def read_csv_data(contents, filename):
    "Read a file with one column of data and returns it."
    df = parse_content(contents, filename)
    return df.iloc[:, 0].values


@add_success_output
def read_csv_fit(contents, filename):
    "Read a file with one column for a fit and return a state handler."
    df = parse_content(contents, filename)
    states_averages = df.iloc[:, 0]
    handler = HMM_State_Handler.from_fit(states_averages)
    # TODO: data_handling.from_fit
    raise NotImplementedError
    return handler


@add_success_output
def read_csv_fit_data(...):
    "Read a csv containing columns for both data and a fit."
    # TODO:
    # - Create the handler
    # - Set the parameters of the Gaussian ditributions
    raise NotImplementedError
