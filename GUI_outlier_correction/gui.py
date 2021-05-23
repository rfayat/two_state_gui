import plotly.graph_objs as go
from .simulation import Data_Simulator
from .data_handling import HMM_State_Handler

N_POINTS = 100000

if __name__ == "__main__":
    # Simulate data
    simulator = Data_Simulator.simulate(n_points=N_POINTS)

    # Create the handler for the states
    handler = HMM_State_Handler.from_parameters(mu_all=simulator.mu_all,
                                          sigma_all=simulator.sigma_all)
    handler.add_fitted_states(simulator.states)

    # Plot the result
    interval_states = handler.get_intervals_states().repeat(2)
    line_data = go.Line(x=handler.time, y=simulator.data)
    line_states = go.Line(x=handler.get_intervals_time().flatten(),
                          y=handler.mu_all[interval_states])
    fig = go.Figure([line_data, line_states])
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
