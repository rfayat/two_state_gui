"""Graphical User Interface for data and two-state model visualization.

Author: Romain Fayat, May 2021
"""
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# import dash_table
import numpy as np
import dash_bootstrap_components as dbc
from .simulation import Data_Simulator
from .data_handling import HMM_State_Handler
from . import file_io
from functools import wraps
from pathlib import Path
from datetime import datetime
import traceback
# Initial value for handler, overriden when adding a fit
handler = None
# Initial value for data, overriden when adding data
data = None
# Hz, overriden by the values stored by the state handler when adding a fit
sr = 30.
# Export folder
export_folder = Path("~/Documents/two_state_gui_exports").expanduser().absolute()
# Export subfolder
path_out = None
# Filename
last_uploaded_file = ""

COLORS = {"navbar": "#17A2B8",
          "plotly": "#119DFF",
          "data": "#636EFA",
          "fit": "#F0644D",
          "corrected": "#27D3A6",
          "ignored": "Black"}


def update_traces(fig, data, handler=None):
    "Update the traces with the raw and corrected data."
    fig = update_data_traces(fig, data)
    if handler is not None:
        fig = update_fit_traces(fig, handler)
        fig = update_corrected_traces(fig, handler)
    else:
        fig = cleanup_fit_traces(fig)
    return fig


def update_data_traces(fig, data):
    "Update the data trace of the figure"
    global sr
    # Trace the raw data
    fig.update_traces(x=np.arange(len(data)) / sr, y=data,
                      selector={"name": "Data"})
    return fig


def update_fit_traces(fig, handler):
    "Update the fit trace of the figure"
    global sr
    if handler is None:
        return fig
    sr = handler.sr
    # Trace a compressed version of the initial fit
    idx_to_plot = np.c_[handler.intervals_start,
                        handler.intervals_end - 1].flatten()

    fig.update_traces(x=handler.time[idx_to_plot],
                      y=handler.get_mu(handler.states[idx_to_plot]),
                      selector={"name": "Fit"})
    return fig


def update_corrected_traces(fig, handler):
    "Update the traces for corrected and ignored data."
    # Create a subset of points to plot to make the figure update faster
    to_plot = np.zeros(handler.n_points, dtype=bool)
    to_plot[::int(2 * handler.sr)] = True  # One clickable point every 2s
    # Always plot the start and stop of intervals
    to_plot[handler.intervals_start] = True
    to_plot[handler.intervals_end - 1] = True

    # Add custom data for hover informations and callbacks
    intervals_number = np.repeat(np.arange(handler.n_intervals),
                                 handler.intervals_durations)
    customdata = np.c_[
        intervals_number,
        handler.states,
        handler.states_corrected,
        handler.time[handler.intervals_start[intervals_number]],
        handler.time[handler.intervals_end[intervals_number] - 1]
    ]
    # Plot the corrected fit
    fig.update_traces(
        x=handler.time[to_plot],
        y=handler.get_mu(handler.states_corrected)[to_plot],
        customdata=customdata[to_plot],
        selector={"name": "Corrected"},
    )
    # Create an array with zero for ignored values and nan elsewhere
    ignored_to_plot = np.full(handler.n_points, np.nan, dtype=float)
    ignored_to_plot[handler.states_corrected == -1] = 0.

    # Plot the ignored data
    fig.update_traces(
        x=handler.time[::int(handler.sr / 2)],
        y=ignored_to_plot[::int(handler.sr / 2)],
        customdata=customdata,
        selector={"name": "Ignored"},
    )
    return fig


def cleanup_fit_traces(fig):
    "Erase all existing fit traces."
    for name in ["Fit", "Corrected", "Ignored"]:
        fig.update_traces(
            x=[],
            y=[],
            selector={"name": name},
        )
    return fig


# Generate the figure
fig = go.Figure()
# Add the data trace using scattergl (much faster than scatter)
trace_data = go.Scattergl(line={"color": COLORS["data"]},
                          mode="lines",
                          name="Data",
                          hoverinfo="skip")

trace_fit_corrected = go.Scattergl(line={"color": COLORS["corrected"],
                                         "width": 3.},
                                   mode="lines", name="Corrected")
trace_ignored = go.Scattergl(marker_symbol="line-ew",
                             marker_line_width=4.,
                             marker_size=5,
                             line={"color": COLORS["ignored"]},
                             mode="markers", name="Ignored")
trace_fit = go.Scattergl(line={"color": COLORS["fit"]},
                         mode="lines", name="Fit", hoverinfo="skip")
fig.add_trace(trace_data)
fig.add_trace(trace_fit)
fig.add_trace(trace_fit_corrected)
fig.add_trace(trace_ignored)
fig.update_xaxes(rangeslider_visible=True)


# Create the app and the layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Two-State GUI"

# Top navigation bar
navbar = dbc.Navbar([
        html.A(
            [dbc.NavbarBrand("Two-State GUI")]
        ),
    ],
    color=COLORS["navbar"],
    dark=True,
)

# Radio buttons
radio_action = dbc.Card([
    dbc.CardBody([
        html.H5("Action on click", className="card-title"),
        dbc.RadioItems(
            options=[
                {"label": "Toggle State", "value": "toggle"},
                {"label": "Ignore Values", "value": "discard"},
            ],
            id="radio_action",
            value="toggle",
        )
    ]),
], style={"width": "18rem"})


def create_button_modal(prop_name, button_str, body_content=[]):
    "Create button linked to a modal warning about data overwrite."
    button = dbc.Button(button_str, id=f"open-modal-{prop_name}",
                        color="info", className="mb-3")
    body = html.Div(body_content),
    modal = dbc.Modal(
        [
            dbc.ModalHeader(button_str),
            dbc.ModalBody(
                body
            ),
            dbc.ModalFooter(
                dbc.ButtonGroup([
                    dbc.Button("Close", id=f"close-modal-{prop_name}"),
                    dbc.Button(button_str, id=f"button-{prop_name}",
                               color="info"),
                ], className="ml-auto"),
            ),
        ],
        id=f"modal-{prop_name}",
        centered=True,
    )
    return button, modal


def create_field_upload(id_upload, id_output):
    upload = dcc.Upload(
        id=id_upload,
        children=html.Div([
        'Drag and Drop or ',
        html.A('Select File')
        ]),
        style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
         },
        # Allow only one file to be uploaded
        multiple=False
    )
    output = html.Div(id=id_output)
    return [upload, output]


warning_overwrite_str = "Warning, this action will overwrite existing data"

button_simulation, modal_simulation = create_button_modal(
    "simulation", "Simulate data", [warning_overwrite_str],
)
button_upload_csv, modal_upload_csv = create_button_modal(
    "upload_csv", "Upload CSV",
    [warning_overwrite_str,
     *create_field_upload('upload_csv', 'output-upload_csv')]
)

button_upload_fit, modal_upload_fit = create_button_modal(
    "upload_fit", "Upload fit",
    [warning_overwrite_str,
     *create_field_upload('upload_fit', 'output-upload_fit')]
)

button_export_csv, modal_export_csv = create_button_modal(
    "export_csv", "Export to CSV",
    ["Export fit and summary to CSV.",
     " Path to the export directory:",
     html.Div([], id="path_export_csv"),
     html.Div([], id="validation_export_csv")]
)


button_fit_hmm, modal_fit_hmm = create_button_modal(
    "fit_hmm", "Fit HMM",
    [warning_overwrite_str,
     html.Div([], id="status-fit_hmm")]
)

# Buttons for I/O and HMM fit
button_group_hmm = dbc.ButtonGroup([
    button_simulation,
    button_upload_csv,
    button_upload_fit,
    button_export_csv,
    dbc.Button("Compute fit", id="collapse-button",
               className="mb-3", color="info"),
], className="mt-6 ml-3")

modal_group_hmm = html.Div([
    modal_simulation,
    modal_upload_csv,
    modal_upload_fit,
    modal_export_csv,
    modal_fit_hmm,
])

# Collapse for setting the HMM parameters
collapse_hmm = dbc.Collapse(
    dbc.Card(
        dbc.CardBody(
            html.Div([
                dbc.Row([
                    dbc.InputGroup([
                        dbc.InputGroupAddon("param1",
                                            addon_type="prepend"),
                        dbc.Input(placeholder="param1",
                                  type="number")],
                                  className="mb-1",
                    ),
                    dbc.InputGroup([
                        dbc.InputGroupAddon("param2",
                                            addon_type="prepend"),
                        dbc.Input(placeholder="param2",
                                  type="number")],
                        className="mb-1",
                    ),
                ]),
                dbc.Row([button_fit_hmm]),
            ]),
        ),
        className="border-0"
    ),
    id="collapse",
)


# Global layout of the app
app.layout = html.Div([
    navbar,
    dcc.Graph(id="data_graph", figure=fig),
    dbc.Row([
        dbc.Col(radio_action, width={"offset": 1}),
        dbc.Col(
            html.Div([
                button_group_hmm,
                collapse_hmm,
                modal_group_hmm,
            ]),
            width={"size": 4, "align": "end"}
        ),
    ], justify="between")
])


@app.callback(Output("collapse", "is_open"),
              [Input("collapse-button", "n_clicks")],
              [State("collapse", "is_open")])
def toggle_collapse(n, is_open):
    """Collapse the HMM fit section."""
    if n:
        return not is_open
    return is_open


def run_simulation(fig):
    "Generate data and update the graph."
    global handler
    global data

    # Simulate data
    N_POINTS = 100000
    simulator = Data_Simulator.simulate(n_points=N_POINTS, mu_all=[.2, .4])
    # Change the value of the global variable data
    data = simulator.data
    # Simulate missing states
    simulated_states = simulator.states
    simulated_states[int(N_POINTS / 2):int(N_POINTS / 2) + int(N_POINTS / 20)] = -1  # noqa E501
    # Create the handler for the states
    handler = HMM_State_Handler.from_parameters(
        mu_all=simulator.mu_all, sigma_all=simulator.sigma_all
    )
    handler.add_fitted_states(simulated_states)
    fig = update_traces(fig, simulator.data, handler)
    fig['layout']['xaxis'].update(range=[handler.time[0], handler.time[-1]])
    return fig


def upload_csv(fig, contents, filename):
    "Upload a csv with data and update the figure."
    data_from_csv, success = file_io.read_csv_data(contents, filename)
    if success:
        global data
        global last_uploaded_file
        last_uploaded_file = filename
        data = data_from_csv
        fig = update_traces(fig, data)
    return fig


def upload_fit(fig, contents, filename):
    "Upload a csv with fitted values and update the figure."
    global handler
    handler, success = file_io.read_csv_fit(contents, filename)
    if success:
        fig = update_fit_traces(fig, handler)
        fig = update_corrected_traces(fig, handler)
    return fig


@app.callback(Output('output-upload_csv', 'children'),
              Input('upload_csv', 'contents'),
              State('upload_csv', 'filename'))
def update_output_upload_csv(contents, filename):
    if contents is not None:
        children = dbc.Alert([
            f"Successfully selected {filename} !\nPress ",
            dbc.Badge("Upload CSV", color="info", className="ml-1"),
            " to overwrite the plot."], color="success")
        return children


@app.callback(Output('output-upload_fit', 'children'),
              Input('upload_fit', 'contents'),
              State('upload_fit', 'filename'))
def update_output_upload_fit(contents, filename):
    if contents is not None:
        children = dbc.Alert([
            f"Successfully selected {filename} !\nPress ",
            dbc.Badge("Upload Fit", color="info", className="ml-1"),
            " to overwrite the plot."], color="success")
        return children


@app.callback(
    Output("modal-simulation", "is_open"),
    [Input("open-modal-simulation", "n_clicks"),
     Input("close-modal-simulation", "n_clicks")],
    [State("modal-simulation", "is_open")])
def toggle_modal_simulation(n1, n2, is_open):
    "Open the modal for running a new simulation."
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-upload_csv", "is_open"),
    [Input("open-modal-upload_csv", "n_clicks"),
     Input("close-modal-upload_csv", "n_clicks")],
    [State("modal-upload_csv", "is_open")])
def toggle_modal_upload_csv(n1, n2, is_open):
    "Open the modal for uploading a new csv."
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-fit_hmm", "is_open"),
    [Input("open-modal-fit_hmm", "n_clicks"),
     Input("close-modal-fit_hmm", "n_clicks")],
    [State("modal-fit_hmm", "is_open")])
def toggle_modal_fit_hmm(n1, n2, is_open):
    "Open the modal for fitting a HMM."
    if n1 or n2:
        return not is_open
    return is_open


def fit_hmm(fig):
    "Fit a HMM and return the updated figure."
    global data
    global handler
    global sr
    handler = HMM_State_Handler(sr)
    handler.fit_predict(data)
    fig = update_fit_traces(fig, handler)
    fig = update_corrected_traces(fig, handler)
    return fig


@app.callback(
    Output("modal-upload_fit", "is_open"),
    [Input("open-modal-upload_fit", "n_clicks"),
     Input("close-modal-upload_fit", "n_clicks")],
    [State("modal-upload_fit", "is_open")])
def toggle_modal_upload_fit(n1, n2, is_open):
    "Open the modal for uploading a new fit."
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-export_csv", "is_open"),
    Output("path_export_csv", "children"),
    Input("open-modal-export_csv", "n_clicks"),
    Input("close-modal-export_csv", "n_clicks"),
    State("modal-export_csv", "is_open"))
def toggle_modal_export_csv(n1, n2, is_open):
    "Open the modal for exporting data to csv."
    global last_uploaded_file
    global export_folder
    global path_out
    path_out = get_output_path(export_folder, last_uploaded_file)
    if n1 or n2:
        return not is_open, str(path_out)
    return is_open, str(path_out)


def get_output_path(path_dir, filename):
    "Return the path to the output directory for csv export."
    path_dir = Path(path_dir)
    # Create the name to the output folder
    now = datetime.now()
    now_str = f"{now.year}{now.month:02d}{now.day:02d}"
    filename = filename.split(".")[0]
    filename = filename + "_" if filename != "" else ""
    # Add a number at the end of the folder name and increment it if needed
    i = 0
    output_dir = path_dir / f"{filename}{now_str}_{i:03d}"
    while output_dir.exists():
        i += 1
        output_dir = path_dir / f"{filename}{now_str}_{i:03d}"
    return output_dir


@app.callback(Output("validation_export_csv", "children"),
              Input("button-export_csv", "n_clicks"))
def export_csv(n_clicks):
    "Export data to csv."
    global path_out
    global handler
    global data
    if n_clicks is None:
        return None
    if handler is None:
        return dbc.Alert("No data available...", color="warning")
    # Create the output directory
    path_out.mkdir(parents=True, exist_ok=True)
    # Create the summary dataframes
    df_data = handler.to_dataframe()
    df_intervals = handler.to_intervals_dataframe()
    df_summary = handler.summary(data)
    # Save the summary dataframes
    df_data.to_csv(path_out / (path_out.name + "_data.csv"),
                   encoding="utf-8", index=False)
    df_intervals.to_csv(path_out / (path_out.name + "_intervals.csv"),
                        encoding="utf-8", index=False)
    df_summary.to_csv(path_out / (path_out.name + "_summary.csv"),
                      encoding="utf-8", index=True)
    return dbc.Alert("Success !", color="success")


@app.callback(Output("data_graph", "figure"),
              Output("status-fit_hmm", "children"),
              Input("data_graph", "clickData"),
              Input("button-simulation", "n_clicks"),
              Input("button-upload_csv", "n_clicks"),
              Input("button-upload_fit", "n_clicks"),
              Input("button-fit_hmm", "n_clicks"),
              State("radio_action", "value"),
              State('upload_csv', 'contents'),
              State('upload_csv', 'filename'),
              State('upload_fit', 'contents'),
              State('upload_fit', 'filename'))
def figure_callback(
    clickData, click_simulate, click_upload_csv, click_upload_fit,
    click_fit_hmm, action, contents_upload_csv, filename_upload_csv,
    contents_upload_fit, filename_upload_fit
):
    """Handle all callbacks affecting the output figure.

    WARNING: Dash doesn't apparently support multiple callbacks with figure
    as output hence this unique handler for all callbacks.
    """
    global fig
    # Ids object whose status has been changed
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # Select the action to perform based on the inputs and changed_id
    trigger_change_interval = clickData is not None and action is not None and changed_id == "data_graph.clickData"  # noqa E501
    trigger_simulation = changed_id == "button-simulation.n_clicks"
    trigger_upload_csv = changed_id == "button-upload_csv.n_clicks"
    trigger_upload_fit = changed_id == "button-upload_fit.n_clicks"
    trigger_fit_hmm = changed_id == "button-fit_hmm.n_clicks"
    if trigger_change_interval:
        return callback_figure_clicked(fig, action, clickData), None
    elif trigger_simulation and click_simulate is not None:
        return run_simulation(fig), None
    elif trigger_upload_csv and click_upload_csv is not None:
        return upload_csv(fig, contents_upload_csv, filename_upload_csv), None
    elif trigger_upload_fit and click_upload_fit is not None:
        return upload_fit(fig, contents_upload_fit, filename_upload_fit), None
    elif trigger_fit_hmm and click_fit_hmm is not None:
        try:
            return fit_hmm(fig), dbc.Alert("Success !", color="success")
        except Exception as e:
            print("Error during HMM fit")
            print(e)
            traceback.print_tb(err.__traceback__)
            return fig, dbc.Alert("Fitting failed...", color="danger")
    else:
        return fig, None


def callback_figure_clicked(fig, action, clickData):
    "Handle figure update when a segment is clicked."
    global handler
    # Get the interval that was clicked
    interval_idx = int(clickData["points"][0]["customdata"][0])
    # Toggle the state of the interval or discard depending on the inputs
    if action == "toggle":
        handler.change_interval_state(interval_idx)
    else:
        handler.change_interval_missing_status(interval_idx)

    # Update the traces and return the updated figure
    fig = update_corrected_traces(fig, handler)
    # Redraw using a time interval around the change
    interval_start_time = clickData["points"][0]["customdata"][3]
    interval_end_time = clickData["points"][0]["customdata"][4]
    time_range = [interval_start_time - 60, interval_end_time + 60]
    fig['layout']['xaxis'].update(range=time_range)
    return fig


if __name__ == "__main__":
    app.run_server(host="127.0.0.1", debug=True)
