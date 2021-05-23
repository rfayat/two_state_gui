"""Code for handling data and manipulating the output of the HMM

Author: Romain Fayat, May 2021
"""
import numpy as np
import pandas as pd
import scipy
import scipy.stats


class Gaussian():
    "A 1D-Gaussian distribution"

    def __init__(self, mu=None, sigma=None):
        "Create a gaussian distribution object with input mean and std"
        self.mu = mu
        self.sigma = sigma

    def set_parameters(self, mu, sigma):
        "Set the parameters of the Gaussian"
        self.mu = mu
        self.sigma = sigma

    @property
    def random_variable(self):
        "Return a scipy.stats.norm object using self's parameters"
        return scipy.stats.norm(loc=self.mu, scale=self.sigma)

    def __repr__(self):
        "Display the parameters of the distribution"
        return ("Gaussian distribution with parameters:",
                f"   μ = {self.mu}",
                f"   σ = {self.sigma}")

    def __call__(self, x):
        "Return the probability of input values under the distribution"
        return self.random_variable.pdf(x)



class HMM_State_Handler():
    "Handler of manual corrections to an HMM fit"

    def __init__(self, n_states=2, sr=30.):
        """Create a handler for manual corrections to an HMM fit

        Data is modelled as having a gaussian distribution whose parameters
        are specific to each hidden state.

        Inputs
        ------
        n_states : int (default = 2)
            Number of hidden states in the HMM

        sr : float (default = 30.)
            Sampling  rate of the time series, in Herz.
        """
        self.n_states = n_states
        self.sr = sr

        # Distribution of the data generated in each state
        self.distributions = [Gaussian() for _ in range(self.n_states)]

    @classmethod
    def from_parameters(cls, mu_all, sigma_all, **kwargs):
        "Instanciate the object from parameters for the gaussian distributions"
        # Make sure that the input lengths match
        assert len(mu_all) == len(sigma_all)
        kwargs.update({"n_states": len(mu_all)})
        # Create the object and set the parameters of the distributions
        self = cls(**kwargs)
        for i, d in enumerate(self.distributions):
            d.set_parameters(mu_all[i], sigma_all[i])

        return self

    @property
    def mu_all(self):
        "Return the mean of the distribution for each state"
        mu_all = [d.mu for d in self.distributions]
        return np.array([e if e is not None else np.nan for e in mu_all])

    @property
    def sigma_all(self):
        "Return the std of the distribution for each state"
        sigma_all = [d.sigma for d in self.distributions]
        return np.array([e if e is not None else np.nan for e in sigma_all])

    @property
    def states_unique(self):
        "Values that can be taken by the states. -1 for missing values"
        return np.arange(-1, self.n_states, dtype=np.int)

    def add_fitted_states(self, states):
        "Add a fitted states time series of length n_points"
        # Replace missing values by -1 and make sure states is an array of int
        states = np.where(np.isnan(states), -1, states).astype(np.int)
        self.states = states
        # Sanity check on the values
        assert np.all(np.isin(states, self.states_unique))
        # Create the array for corrected states values
        self.states_corrected = self.states.copy()

    def get_mu(self, states_indexes):
        "Mean of the distribution matching the input states (nan when missing)"
        return np.append(self.mu_all, np.nan)[states_indexes]

    def get_sigma(self, states_indexes):
        "Std of the distribution matching the input states (nan when missing)"
        return np.append(self.sigma_all, np.nan)[states_indexes]

    @property
    def n_points(self):
        "Return the number of points in the states time series"
        return len(self.states)

    @property
    def time(self):
        "Return an array of time values"
        return np.arange(self.n_points) / self.sr

    def as_dataframe(self):
        "Return the states time series as a pandas dataframe"
        return pd.DataFrame({"time": self.time,
                             "states": self.states,
                             "states_corrected": self.states_corrected})

    def get_changepoint(self):
        """Return an array for states changepoints indexes.

        The indexes correspond to the point immediately following changepoints.
        The first changepoint is 0 by convention.
        """
        is_changepoint = self.states[:-1] != self.states[1:]
        changepoint_indexes =  np.argwhere(is_changepoint).flatten() + 1
        return np.append(0, changepoint_indexes)

    def get_changepoint_time(self):
        "Return an array of time where a change in state occured"
        return self.time[self.get_changepoint()]

    def get_intervals(self):
        "Get the first and last index of intervals with constant state"
        interval_end = np.append(self.get_changepoint()[1:], self.n_points - 1)
        return np.c_[self.get_changepoint(), interval_end]

    def get_intervals_time(self):
        "Get the first and last timepoints of intervals with constant state"
        return self.time[self.get_intervals()]

    def get_intervals_states(self):
        "Return an array of the state of each interval"
        return self.states[self.get_changepoint()]


if __name__ == "__main__":
    from .simulation import Data_Simulator
    mu_all = [0, .4]
    sigma_all = [.1, .1]
    simulator = Data_Simulator.simulate(mu_all=mu_all,
                                        sigma_all=sigma_all,
                                        n_points=100000)

    handler = HMM_State_Handler.from_parameters(mu_all=mu_all, sigma_all=sigma_all)
    handler.add_fitted_states(simulator.states)
    handler.mu_all[handler.get_intervals_states().repeat(2)]
