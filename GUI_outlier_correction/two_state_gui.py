import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import dash_bootstrap_components as dbc

import pandas as pd
from GUI_outlier_correction.simulation import Data_Simulator
import numpy as np

colors = {"navbar": "#17A2B8", "plotly": "#119DFF"}

# Simulate data
simulator = Data_Simulator.simulate(n_points=100000,
                                    sigma_all=[.15, .15])
df = simulator.as_dataframe()
# Instead of using states_average directly, use the transitions times
# (Much less points to plot for the same result)
transition_start = (simulator.transitions_times * simulator.sr).astype(int)
transition_stop = transition_start - 1
transition_to_plot = np.c_[transition_stop, transition_start].flatten()[1:]
transition_to_plot = np.clip(transition_to_plot, 0, len(df) - 1)


# Generate the figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.time,
                         y=df.data,
                         mode='lines', name='Data', hoverinfo="skip"))
fig.add_trace(go.Scatter(x=df.time.iloc[transition_to_plot],
                         y=df.states_averages.iloc[transition_to_plot],
                         mode='lines', name='Fit'))

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

# Global layout of the app
app.layout = html.Div([
    navbar,
    dcc.Graph(id='data_graph', figure=fig),
    button_group_action,
    html.Div([
        dbc.ButtonGroup([
            dbc.Button("Upload CSV", color="info", className="mb-3"),
            dbc.Button("Export to CSV", color="info", className="mb-3"),
            dbc.Button("Edit fit parameters", id="collapse-button",
                       className="mb-3", color="info"),
        ], className="mb-3"),
        collapse_hmm
    ]),
])

# Interactive elements
@app.callback(Output("collapse", "is_open"),
              [Input("collapse-button", "n_clicks")],
              [State("collapse", "is_open")])
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', debug=True)
