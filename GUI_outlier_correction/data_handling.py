"""Code for handling data and manipulating the output of the HMM.

Author: Romain Fayat, May 2021
"""
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from .hmm import fit_hmm, HMM, Gaussian
from .helpers import percentile, get_intervals_idx


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
        states = np.where(np.isnan(states), -1, states).astype(int)
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

    @classmethod
    def from_parameters(cls, mu_all, sigma_all=None, states=None, **kwargs):
        "Instantiation from states and states parameters."
        self = super(HMM_State_Handler, cls).from_parameters(mu_all, sigma_all, **kwargs)
        if states is not None:
            self.add_fitted_states(states)
        return self

    @classmethod
    def from_fit(cls, states_averages, **kwargs):
        "Instantiation from an array of Gaussians' means."
        states = np.zeros(len(states_averages), dtype=int)
        mu_all = np.unique(states_averages)

        # Deal with missing fit values
        if np.any(np.isnan(states_averages)):
            states[np.isnan(states_averages)] = -1
            mu_all = mu_all[~np.isnan(mu_all)]

        for i, mu in enumerate(mu_all):
            states[np.isclose(states_averages, mu)] = i

        return cls.from_parameters(mu_all=mu_all, states=states, **kwargs)

    def fit_predict(self, data, ignore_data=None, **kwargs):
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

    def to_dataframe(self):
        "Create a pandas dataframe with the states and corrected states."
        df = pd.DataFrame(dict(
            time=self.time,
            states=self.states,
            states_corrected=self.states_corrected,
            states_mean = self.mu_all[self.states],
            states_corrected_mean = self.mu_all[self.states_corrected],
        ))
        return df

    def to_intervals_dataframe(self):
        "Create a summary dataframe with one row for each window."
        df = pd.DataFrame(dict(
            intervals_idx=np.arange(self.n_intervals),
            intervals_states=self.intervals_states,
            intervals_states_corrected=self.intervals_states_corrected,
            intervals_start_bins=self.intervals_start,
            intervals_end_bins=self.intervals_end,
            intervals_duration_bins=self.intervals_durations,
            intervals_start_time=self.intervals_start / self.sr,
            intervals_end_time=self.intervals_end / self.sr,
            intervals_duration_time=self.intervals_durations / self.sr,
            intervals_mean=self.mu_all[self.intervals_states],
            intervals_mean_corrected=self.mu_all[self.intervals_states_corrected],  # noqa E501
        ))
        return df

    def summary(self, data=None):
        "Create a pandas dataframe with a summary of the corrected states."
        # Get the corrected intervals indexes
        # N.B. Intervals merged by a manual correction are now treated as
        # only one.
        states_corrected_start, states_corrected_end = get_intervals_idx(self.states_corrected)  # noqa E501
        states_corrected_duration = states_corrected_end - 1 - states_corrected_start  # noqa E501
        states_corrected_duration[0] += 1
        n_intervals_corrected = len(states_corrected_duration)
        intervals_corrected_idx = np.repeat(np.arange(n_intervals_corrected),
                                            states_corrected_duration)
        # Create a consolidated dataframe with the corrected states and data
        df = pd.DataFrame(dict(
            states_corrected=self.states_corrected,
            intervals_corrected_idx=intervals_corrected_idx,
            time=self.time,
            duration=1 / self.sr,
            data=data,
        ))
        # Compute statistics on the durations of the intervals
        agg_functions_duration = [
            "min", percentile(25), np.median,
            percentile(75), "max", "mean", "std", "count"
        ]
        summary_duration = df.groupby(by="intervals_corrected_idx").agg({
            "duration": "sum",
            "states_corrected": "first",
        }).groupby(by="states_corrected").agg({
            "duration": agg_functions_duration
        })
        try:
            # Compute some additional statistics about the data in each state
            agg_functions_data = ["mean", "std"]
            summary = df.groupby(by="states_corrected").agg({
                "data": agg_functions_data
            }).join(summary_duration)
            # Return the dataframe summarizing all computed stats
            return summary
        # In case no data was provided (thus raising a DataError), return the
        # summary only for the durations
        except pd.core.groupby.groupby.DataError:
            return summary_duration


if __name__ == "__main__":
    from .simulation import Data_Simulator
    mu_all = [-1, .4]
    sigma_all = [.1, .1]
    simulator = Data_Simulator.simulate(mu_all=mu_all,
                                        sigma_all=sigma_all,
                                        n_points=100000)
    simulated_states = simulator.states
    simulated_states[10000:15000] = -1
    handler = HMM_State_Handler.from_parameters(
        mu_all=mu_all, sigma_all=sigma_all, states=simulated_states
    )
    print(handler.mu_all[handler.intervals_states])
