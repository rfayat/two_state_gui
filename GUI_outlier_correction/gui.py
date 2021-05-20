import plotly.graph_objs as go
from .simulation import Data_Simulator

N_POINTS = 100000

if __name__ == "__main__":
    simulator = Data_Simulator.simulate(n_points=N_POINTS)
    df = simulator.as_dataframe()
    line_data = go.Line(x=df.time, y=df.data)
    line_states = go.Line(x=df.time, y=df.states_averages)
    fig = go.Figure([line_data, line_states])
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
