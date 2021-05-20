"""Tools for simulating raw data traces and HMM fits.

Author: Romain Fayat, May 2021
"""
import numpy as np


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

    def simulate(self):
        "Create a simulation of the state transitions and data"
        self.states = self.simulate_state_transitions()
        data = np.zeros(self.n_points)

        # Loop over the states and generate data from each of them
        for s, (mu, sigma) in enumerate(zip(self.mu_all, self.sigma_all)):
            n_points_state = np.sum(self.states == s)
            # Use a gaussian distribution for the data
            data_state = np.random.normal(mu, sigma, n_points_state)
            data[self.states == s] = data_state

        self.data = data
        return self.time, self.states, self.data

    def simulate_state_transitions(self):
        "Create state transitions using exponential time intervals"
        cumulated_time = 0.
        transitions_times = np.array([])
        # Create transition times until we reach the end of the duration
        while cumulated_time < self.n_points / self.sr:

            t = np.random.exponential(self.tau)
            cumulated_time += t
            transitions_times = np.append(transitions_times, cumulated_time)

        # Truncate the transition times to get an even number of transitions
        if len(transitions_times) % 2 != 0:
            transitions_times = transitions_times[:-1]

        # Use the transition times to fill the states array
        states = np.zeros(self.n_points, dtype=np.int)
        for start_time, stop_time in transitions_times.reshape((-1, 2)):
            states[(self.time >= start_time) & (self.time < stop_time)] = 1

        return states


if __name__ == "__main__":
    # Show an example trace
    import matplotlib.pyplot as plt
    time, states, data = Data_Simulator(n_points=100000).simulate()
    fig, ax = plt.subplots()
    ax.plot(time, data)
    fig.show()
    str(input("Press Enter to quit"))
