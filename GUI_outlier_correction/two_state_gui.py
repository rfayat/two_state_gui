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


COLORS = {"navbar": "#17A2B8",
          "plotly": "#119DFF",
          "data": "#636EFA",
          "fit": "#F0644D",
          "corrected": "#27D3A6",
          "ignored": "Black"}


def update_traces(fig, handler, data):
    "Update the traces with the raw and corrected data."
    # Trace the raw data
    fig.update_traces(x=handler.time, y=data,
                      selector={"name": "Data"})
    # Trace a compressed version of the initial fit
    idx_to_plot = np.c_[handler.intervals_start,
                        handler.intervals_end - 1].flatten()
    fig.update_traces(x=handler.time[idx_to_plot],
                      y=handler.get_mu(handler.states[idx_to_plot]),
                      selector={"name": "Fit"})
    fig = update_corrected_traces(fig, handler)  # Plot the corrected state
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
                dbc.Row([dbc.Button("Fit HMM",
                                    color="success",
                                    className="mb-1")]),
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

# Buttons for I/O and HMM fit
button_group_hmm = dbc.ButtonGroup([
    dbc.Button("Simulate data", id="simulate-button",
               color="info", className="mb-3"),
    dbc.Button("Upload CSV", color="info", className="mb-3"),
    dbc.Button("Export to CSV", color="info", className="mb-3"),
    dbc.Button("Import fit", color="info", className="mb-3"),
    dbc.Button("Compute fit", id="collapse-button",
               className="mb-3", color="info"),
], className="mt-6 ml-3")

# Global layout of the app
app.layout = html.Div([
    navbar,
    dcc.Graph(id="data_graph", figure=fig),
    dbc.Row([
        dbc.Col(radio_action, width={"offset": 1}),
        dbc.Col(
            html.Div([
                button_group_hmm,
                collapse_hmm
            ]),
            width={"size": 4, "align": "end"}
        ),
    ], justify="between")
])


# # Interactive elements
# def preserve_xrange(f):
#     "Make sure the x axis' range is maintained after calling the function."
#     @wraps(f)
#     def g(*args, **kwargs):
#         global fig
#         # TODO: not working, even when using fig.full_figure_for_development
#         # Grab the x axis' range before running the function
#         xrange_before = fig.layout.xaxis.range  # noqa W0612
#         # Run the function
#         out = f(*args, **kwargs)
#         # Set the x axis' range to its initial value
#         # TODO
#         return out
#     return g


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
    fig = update_traces(fig, handler, simulator.data)
    fig['layout']['xaxis'].update(range=[handler.time[0], handler.time[-1]])
    return fig


@app.callback(Output("data_graph", "figure"),
              Input("data_graph", "clickData"),
              Input("simulate-button", "n_clicks"),
              State("radio_action", "value"))
def change_interval_state(clickData, simulate_click, action):
    "Handle callbacks affecting the output figure."
    global fig
    global handler
    # WARNING: Dash doesn't apparently support multiple callbacks with figure
    # as output hence this very circumvolute implementation...
    # Ids object whose status has been changed
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # Select the action to perform based on the inputs and changed_id
    trigger_change_interval = clickData is not None and action is not None and changed_id == "data_graph.clickData"  # noqa E501
    trigger_simulation = changed_id == "simulate-button.n_clicks" and simulate_click is not None  # noqa 501

    if trigger_change_interval:
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
    elif trigger_simulation:  # noqa E501
        return run_simulation(fig)
    else:
        return fig


if __name__ == "__main__":
    app.run_server(host="127.0.0.1", debug=True)
