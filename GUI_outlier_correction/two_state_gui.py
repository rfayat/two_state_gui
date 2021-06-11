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
from functools import wraps
# Initial value for handler, overriden when adding a fit
handler = None
# Hz, overriden by the values stored by the state handler when adding a fit
sr = 30.

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
        fig = cleanup_fit_traces()
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
    to_plot = np.zeros(handler.n_points, dtype=np.bool)
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
    ignored_to_plot = np.full(handler.n_points, np.nan, dtype=np.float)
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
                dbc.Row([dbc.Button("Fit HMM",
                                    color="success",
                                    className="mb-1")]),
            ]),
        ),
        className="border-0"
    ),
    id="collapse",
)

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


def create_data_ovewrite_modal(prop_name, button_str):
    "Create button linked to a modal warning about data overwrite."
    button = dbc.Button(button_str, id=f"open-modal-{prop_name}",
                        color="info", className="mb-3")
    modal = dbc.Modal(
        [
            dbc.ModalHeader(button_str),
            dbc.ModalBody(
                "Warning, this action will overwrite existing data"
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

button_simulation, modal_simulation = create_data_ovewrite_modal(
    "simulation", "Simulate data"
)
button_upload_csv, modal_upload_csv = create_data_ovewrite_modal(
    "upload_csv", "Upload CSV"
)

# Buttons for I/O and HMM fit
button_group_hmm = dbc.ButtonGroup([
    button_simulation,
    button_upload_csv,
    dbc.Button("Export to CSV", color="info", className="mb-3"),
    dbc.Button("Import fit", color="info", className="mb-3"),
    dbc.Button("Compute fit", id="collapse-button",
               className="mb-3", color="info"),
], className="mt-6 ml-3")

modal_group_hmm = html.Div([
    modal_simulation,
    modal_upload_csv,
])

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

    # Simulate data
    N_POINTS = 100000
    simulator = Data_Simulator.simulate(n_points=N_POINTS, mu_all=[.2, .4])
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


def upload_csv(fig):
    "Upload a csv with data and update the figure."
    cleanup_fit_traces(fig)
    return fig


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


@app.callback(Output("data_graph", "figure"),
              Input("data_graph", "clickData"),
              Input("button-simulation", "n_clicks"),
              Input("button-upload_csv", "n_clicks"),
              State("radio_action", "value"))
def figure_callback(clickData, click_simulate, click_upload_csv, action):
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
    if trigger_change_interval:
        return callback_figure_clicked(fig, action, clickData)
    elif trigger_simulation and click_simulate is not None:
        return run_simulation(fig)
    elif trigger_upload_csv and  click_upload_csv is not None:
        return upload_csv(fig)
    else:
        return fig


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
