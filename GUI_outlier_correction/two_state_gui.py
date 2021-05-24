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
import json
from functools import wraps


colors = {"navbar": "#17A2B8",
          "plotly": "#119DFF",
          "data": "#636EFA",
          "fit": "#F0644D",
          "corrected": "#27D3A6"}

# Simulate data
N_POINTS = 100000
simulator = Data_Simulator.simulate(n_points=N_POINTS)
# Simulate missing states
simulated_states = simulator.states
simulated_states[int(N_POINTS / 2):int(N_POINTS / 2) + int(N_POINTS / 20)] = -1
# Create the handler for the states
handler = HMM_State_Handler.from_parameters(
    mu_all=simulator.mu_all, sigma_all=simulator.sigma_all
)
handler.add_fitted_states(simulated_states)


# Generate the figure
fig = go.Figure()

# Add the data trace using scattergl (much faster than scatter)
trace_data = go.Scattergl(x=handler.time,
                          y=simulator.data,
                          line={"color": colors["data"]},
                          mode="lines",
                          opacity=.5,
                          name="Data",
                          hoverinfo="skip")
trace_fit_corrected = go.Scattergl(
    x=handler.get_intervals_time().flatten(),
    y=handler.get_mu(handler.intervals_states_corrected.repeat(2)),
    customdata=np.arange(handler.n_intervals).repeat(2),
    line={"color": colors["corrected"]},
    mode="lines", name="Corrected"
)
trace_fit = go.Scattergl(
    x=handler.get_intervals_time().flatten(),
    y=handler.get_mu(handler.intervals_states.repeat(2)),
    line={"color": colors["fit"]},
    mode="lines", name="Fit", hoverinfo="skip"
)
fig.add_trace(trace_data)
fig.add_trace(trace_fit_corrected)
fig.add_trace(trace_fit)
fig.update_xaxes(rangeslider_visible=True)

# Create the app and the layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Collapse for setting the HMM parameters
collapse_hmm = dbc.Collapse(
    dbc.Card(dbc.CardBody(
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
    )),
    id="collapse"
)

# Top navigation bar
navbar = dbc.Navbar([
        html.A(
            [dbc.NavbarBrand("Two-State GUI")]
        ),
    ],
    color=colors["navbar"],
    dark=True,
)

# Buttons for changing the state of data points
button_group_action = dbc.Row([
    dbc.ButtonGroup(
        [dbc.Button("<<",  color="light"),
         dbc.Button("Toggle", color="primary"),
         dbc.Button("Ignore",  color="dark"),
         dbc.Button(">>", color="light")],
        # className="mr-1",
    ),
], justify="center")

# Buttons for I/O and HMM fit
button_group_hmm = dbc.ButtonGroup([
    dbc.Button("Upload CSV", color="info", className="mb-3"),
    dbc.Button("Export to CSV", color="info", className="mb-3"),
    dbc.Button("Import fit", color="info", className="mb-3"),
    dbc.Button("Compute fit", id="collapse-button",
               className="mb-3", color="info"),
], className="mb-3")

# Global layout of the app
app.layout = html.Div([
    navbar,
    dcc.Graph(id="data_graph", figure=fig),
    button_group_action,
    html.Div([
        button_group_hmm,
        collapse_hmm
    ]),
    html.Div([], id="out")
])


# Interactive elements
def preserve_xrange(f):
    "Make sure the x axis' range is maintained after calling the function."
    @wraps(f)
    def g(*args, **kwargs):
        global fig
        # Grab the x axis' range before running the function
        full_fig = fig.full_figure_for_development(warn=False)
        xrange_before = full_fig.layout.xaxis.range
        # Run the function
        out = f(*args, **kwargs)
        # Set the x axis' range to its initial value
        # TODO
        return out
    return g


@app.callback(Output("collapse", "is_open"),
              [Input("collapse-button", "n_clicks")],
              [State("collapse", "is_open")])
def toggle_collapse(n, is_open):
    """Collapse the HMM fit section."""
    if n:
        return not is_open
    return is_open


@app.callback(Output("data_graph", "figure"),
              Input("data_graph", "clickData"))
def change_interval_state(clickData):
    "Change the state of an interval."
    global handler
    global fig
    if clickData is not None:
        interval_idx = clickData["points"][0]["customdata"]
        handler.change_interval_state(interval_idx)
        fig.update_traces(
            x=handler.get_intervals_time().flatten(),
            y=handler.get_mu(handler.intervals_states_corrected.repeat(2)),
            selector={"name": "Corrected"},
        )
    return fig


if __name__ == "__main__":
    app.run_server(host="127.0.0.1", debug=True)
