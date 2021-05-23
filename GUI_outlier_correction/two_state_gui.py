import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import dash_bootstrap_components as dbc

import pandas as pd
from .simulation import Data_Simulator
from .data_handling import HMM_State_Handler
import numpy as np
import json

colors = {"navbar": "#17A2B8", "plotly": "#119DFF"}

# Simulate data
N_POINTS = 100000
simulator = Data_Simulator.simulate(n_points=N_POINTS)

# Create the handler for the states
handler = HMM_State_Handler.from_parameters(mu_all=simulator.mu_all,
                                      sigma_all=simulator.sigma_all)
handler.add_fitted_states(simulator.states)


# Generate the figure
fig = go.Figure()
trace_data = go.Scatter(x=handler.time,
                        y=simulator.data,
                        mode="lines", name="Data", hoverinfo="skip")

trace_fit = go.Scatter(x=handler.get_intervals_time().flatten(),
                       y=handler.get_mu(handler.intervals_states.repeat(2)),
                       mode="lines", name="Fit", hoverinfo="skip")
trace_fit_corrected = go.Scatter(
    x=handler.get_intervals_time().flatten(),
    y=handler.get_mu(handler.intervals_states_corrected.repeat(2)),
    mode="lines", name="Corrected"
)
fig.add_trace(trace_data)
fig.add_trace(trace_fit)
fig.add_trace(trace_fit_corrected)
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
        #className="mr-1",
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
@app.callback(Output("collapse", "is_open"),
              [Input("collapse-button", "n_clicks")],
              [State("collapse", "is_open")])
def toggle_collapse(n, is_open):
    "Collapse the HMM fit section"
    if n:
        return not is_open
    return is_open

@app.callback(Output("out", "children"),
              Input("data_graph", "clickData"))
def display_click_data(clickData):
    out = json.dumps(clickData, indent=2)
    print(out)
    return out



if __name__ == "__main__":
    app.run_server(host="127.0.0.1", debug=True)
