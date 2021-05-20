"""Tools for simulating raw data traces and HMM fits.

Author: Romain Fayat, May 2021
"""
import numpy as np
import pandas as pd
from functools import wraps


def check_fitted(f):
    "Method decorator checking that the object is fitted before running"

    @wraps(f)
    def decorated(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("No data to export, please run the simulation.")
        return f(self, *args, **kwargs)

    return decorated

def get_states_from_transitions_times(transition_times, n_points, sr=30.):
    "Return the state (array of 0 and 1) from transition times"
    states = np.zeros(n_points, dtype=np.int)
    # Convert the transition times to indexes
    transitions_index = (transition_times * sr).astype(np.int)
    # Truncate the transition times to get an even number of transitions
    if len(transitions_index) % 2 != 0:
        states[transitions_index[-1]:] = 1
        transitions_index = transitions_index[:-1]

    # Use the transition times to fill the states array
    for start_idx, stop_idx in transitions_index.reshape((-1, 2)):
        states[max(0, start_idx):min(n_points, stop_idx)] = 1

    return states


def compute_transitions_times(max_t, tau):
    "Compute transition times up to max_t for a given exponential parameter"
    cumulated_time = 0.
    transitions_times = np.array([])
    # Create transition times until we reach the end of the duration
    while cumulated_time < max_t:
        t = np.random.exponential(tau)
        cumulated_time += t
        transitions_times = np.append(transitions_times, cumulated_time)

    return transitions_times


class Data_Simulator():
    "Class for simulating data"

    def __init__(self,
                 mu_all=[0, .4],
                 sigma_all=[.1, .1],
                 tau=10.,
                 n_points=10000,
                 sr=30.):
        """Create the object and store the parameters

        Inputs
        ------
        mu_all, list of floats
            The averages for the two-state gaussian

        sigma_all, list of floats
            The standard deviations for the two-state gaussian

        tau, floats
            The exponential parameter (in seconds) for the interval between
            state transition

        n_points, integer
            The number of points to simulate

        sr, float
            The sampling rate of the data
        """
        self.mu_all = mu_all
        self.sigma_all = sigma_all
        self.tau = tau
        self.n_points = n_points
        self.sr = sr
        self.time = np.arange(self.n_points) / sr
        self._fitted = False

    @classmethod
    def simulate(cls, *args, **kwargs):
        "Create a simulation of the state transitions and data"
        self = cls(*args, **kwargs)
        self.transitions_times = compute_transitions_times(
            self.time[-1], self.tau
        )
        self.states = get_states_from_transitions_times(
            self.transitions_times, self.n_points, self.sr
        )
        data = np.zeros(self.n_points)
        # Loop over the states and generate data from each of them
        for s, (mu, sigma) in enumerate(zip(self.mu_all, self.sigma_all)):
            n_points_state = np.sum(self.states == s)
            # Use a gaussian distribution for the data
            data_state = np.random.normal(mu, sigma, n_points_state)
            data[self.states == s] = data_state

        self.data = data
        self._fitted = True
        return self

    @property
    @check_fitted
    def states_averages(self):
        "Return a time series with the average value of each state"
        return np.where(self.states, self.mu_all[1], self.mu_all[0])

    @check_fitted
    def as_dataframe(self):
        "Return the simulation as a pandas dataframe"
        df = pd.DataFrame({"time": self.time,
                           "data": self.data,
                           "states": self.states,
                           "states_averages": self.states_averages})
        # Convert time from float to timedelta
        # df["time"] = df.time.apply(lambda x: pd.Timedelta(x, unit="S"))
        return df


# %%
if __name__ == "__main__":
    # Show an example trace
    import matplotlib.pyplot as plt
    simulator = Data_Simulator.simulate(n_points=10000)
    df = simulator.as_dataframe()
    fig, ax = plt.subplots()
    ax.plot(df.time, df.data)
    fig.show()
    str(input("Press Enter to quit"))
