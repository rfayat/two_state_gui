"""Code for handling data and manipulating the output of the HMM.

Author: Romain Fayat, May 2021
"""
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from functools import wraps
from .hmm import fit_hmm


def check_fitted(f):
    "Check that the object is fitted before running the decorated method."

    @wraps(f)
    def decorated(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Object was not fitted.")
        return f(self, *args, **kwargs)

    return decorated


class Gaussian():
    "A 1D-Gaussian distribution."

    def __init__(self, mu=None, sigma=None):
        "Create a gaussian distribution object with input mean and std."
        self._fitted = not (mu is None or sigma is None)
        self.mu = mu
        self.sigma = sigma

    def set_parameters(self, mu, sigma):
        "Set the parameters of the Gaussian."
        self.mu = mu
        self.sigma = sigma
        self._fitted = True

    @property
    @check_fitted
    def random_variable(self):
        "Return a scipy.stats.norm object using self's parameters."
        return scipy.stats.norm(loc=self.mu, scale=self.sigma)

    @check_fitted
    def __repr__(self):
        "Display the parameters of the distribution."
        return ("Gaussian distribution with parameters:\n",
                f"   μ = {self.mu}\n",
                f"   σ = {self.sigma}")

    def __call__(self, x):
        "Return the probability of input values under the distribution."
        return self.random_variable.pdf(x)


class HMM():
    "Hidden Markov model with Gaussian emission."

    def __init__(self, n_states=2):
        "Create a HMM with n_states hidden states."
        self.n_states = n_states
        # Distribution of the data generated in each state
        self.distributions = [Gaussian() for _ in range(self.n_states)]
        self._fitted = False

    @classmethod
    def from_parameters(cls, mu_all, sigma_all, **kwargs):
        "Instanciate the object from the gaussians' parameters."
        # Make sure that the input lengths match
        kwargs.update({"n_states": len(mu_all)})
        # Create the object and set the parameters of the distributions
        self = cls(**kwargs)
        self.set_parameters(mu_all, sigma_all)
        return self

    def set_parameters(self, mu_all, sigma_all):
        "Set the parameters of the Gaussian distributions to input values."
        assert len(mu_all) == self.n_states
        assert len(mu_all) == len(sigma_all)
        for i, d in enumerate(self.distributions):
            d.set_parameters(mu_all[i], sigma_all[i])
        self._fitted = True

    def fit_predict(self, data, ignore_data=None, **kwargs):
        """Fit the HMM and return the predicted states.

        Parameters
        ----------
        data : array, shape=(n_samples,)
            The input data

        ignore_data : array of boolean, shape=(n_samples,), default=None
            Array indicating points that will be ignored. If None is provided,
            all points are used for fitting the HMM.

        **kwargs
            Additional key-word argument passed to hmm.fit_hmm

        """
        if ignore_data is None:
            ignore_data = np.zeros(len(data), dtype=np.bool)

        states, mu_all, sigma_all = fit_hmm(data[~ignore_data],
                                            self.n_states,
                                            **kwargs)
        self.set_parameters(mu_all, sigma_all)

        # Set ignored values to -1
        states_with_ignored = np.full(len(data), -1, dtype=np.int)
        states_with_ignored[~ignore_data] = states
        return states_with_ignored

    @property
    def states_unique(self):
        "Values that can be taken by the states. -1 for missing values."
        return np.append(np.arange(self.n_states, dtype=np.int), -1)

    @property
    @check_fitted
    def mu_all(self):
        "Return the mean of the distribution for each state."
        mu_all = [d.mu for d in self.distributions]
        return np.array([e if e is not None else np.nan for e in mu_all])

    @property
    @check_fitted
    def sigma_all(self):
        "Return the std of the distribution for each state."
        sigma_all = [d.sigma for d in self.distributions]
        return np.array([e if e is not None else np.nan for e in sigma_all])

    def get_mu(self, states_indexes):
        "Mean of the distribution matching the input states (nan for missing)."
        return np.append(self.mu_all, np.nan)[states_indexes]

    def get_sigma(self, states_indexes):
        "Std of the distribution matching the input states (nan for missing)."
        return np.append(self.sigma_all, np.nan)[states_indexes]


class HMM_State_Handler(HMM):
    "Handler of manual corrections to an HMM fit."

    def __init__(self, sr=30., *args, **kwargs):
        """Create a handler for manual corrections to an HMM fit.

        Data is modelled as having a gaussian distribution whose parameters
        are specific to each hidden state.

        Inputs
        ------
        sr : float (default = 30.)
            Sampling  rate of the time series, in Herz.

        *args, **kwargs
            Additional arguments passed to HMM
        """
        super().__init__(*args, **kwargs)
        self.sr = sr

    @property
    def time(self):
        "Return an array of time values."
        return np.arange(self.n_points) / self.sr

    def add_fitted_states(self, states):
        "Add a fitted states time series of length n_points."
        # Replace missing values by -1 and make sure states is an array of int
        states = np.where(np.isnan(states), -1, states).astype(np.int)
        # Sanity check on the values
        assert np.all(np.isin(states, self.states_unique))
        # Compte the state changepoints
        is_changepoint = states[:-1] != states[1:]
        changepoints_indexes = np.argwhere(is_changepoint).flatten() + 1
        # Store a sparse representation of the states array
        self.n_points = len(states)
        self.intervals_start = np.append(0, changepoints_indexes)
        self.intervals_states = states[self.intervals_start]
        self.intervals_states_corrected = self.intervals_states.copy()

    def fit(self, data, ignore_data=None, **kwargs):
        """Fit the HMM and handle the parsing of the resulting states.

        Parameters
        ----------
        data : array, shape=(n_samples,)
            The input data

        ignore_data : array of boolean, shape=(n_samples,), default=None
            Array indicating points that will be ignored. If None is provided,
            all points are used for fitting the HMM.

        **kwargs
            Additional key-word argument passed to hmm.fit_hmm

        """
        states = super().fit_predict(data, ignore_data=ignore_data, **kwargs)
        self.add_fitted_states(states)
        return self

    @property
    def n_intervals(self):
        "Return the number of intervals."
        return len(self.intervals_start)

    @property
    def intervals_end(self):
        "Last index for each interval."
        return np.append(self.intervals_start[1:], self.n_points)

    @property
    def intervals_durations(self):
        "Return the intervals' durations."
        return self.intervals_end - self.intervals_start

    @property
    def states(self):
        "States time series as an array of integers."
        return np.repeat(self.intervals_states, self.intervals_durations)

    @property
    def states_corrected(self):
        "States time series as an array of integers."
        return np.repeat(self.intervals_states_corrected,
                         self.intervals_durations)

    def as_dataframe(self):
        "Return the states time series as a pandas dataframe."
        return pd.DataFrame({"time": self.time,
                             "states": self.states,
                             "states_corrected": self.states_corrected})

    def set_interval_to_state(self, interval_idx, corrected_state_number):
        "Set the corrected state number of an interval to an input value."
        assert corrected_state_number in self.states_unique
        self.intervals_states_corrected[interval_idx] = corrected_state_number
        return corrected_state_number

    def change_interval_state(self, interval_idx):
        "Change the corrected state for an interval and return the new value."
        value_corrected_old = self.intervals_states_corrected[interval_idx]
        # Deal with intervals with ignored data
        if value_corrected_old == -1:
            return self.change_interval_missing_status(interval_idx)
        # Replace the old state to the next valid one
        new = range(self.n_states)[(value_corrected_old + 1) % self.n_states]
        return self.set_interval_to_state(interval_idx, new)

    def change_interval_missing_status(self, interval_idx):
        """Toggle the missing status of an interval.

        If the interval has not corrected states -1 set it to -1.
        Else if the initial fit has not state -1, set it to the initial fit.
        Else set it to 0.
        """
        value_corrected_old = self.intervals_states_corrected[interval_idx]
        if value_corrected_old != -1:
            return self.set_interval_to_state(interval_idx, -1)
        else:
            value_fit = self.intervals_states[interval_idx]
            if value_fit != -1:
                return self.set_interval_to_state(interval_idx, value_fit)
            else:
                return self.set_interval_to_state(interval_idx, 0)


if __name__ == "__main__":
    from .simulation import Data_Simulator
    mu_all = [0, .4]
    sigma_all = [.1, .1]
    simulator = Data_Simulator.simulate(mu_all=mu_all,
                                        sigma_all=sigma_all,
                                        n_points=100000)

    handler = HMM_State_Handler.from_parameters(
        mu_all=mu_all, sigma_all=sigma_all
    )
    handler.add_fitted_states(simulator.states)
    print(handler.mu_all[handler.intervals_states])
