"""Functions for fitting an Hidden markov model on data.

Author: Romain Fayat, May 2021
"""
import numpy as np
from functools import wraps
import ssm
import pandas as pd
from .helpers import percentile, get_intervals_idx


def check_fitted(f):
    "Check that the object is fitted before running the decorated method."

    @wraps(f)
    def decorated(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Object was not fitted.")
        return f(self, *args, **kwargs)

    return decorated


def detect_outlier_intervals(data, states, iqr_factor=3.):
    """Detect outliers in intervals fitted on data.

    From the provided states, intervals with consistent state are determined
    in data and the average value of each interval is computed.

    For each state, the quartiles of the averages of the intervals is computed,
    we then use the inter-quartile-range (IQR) to exclude potential outliers.
    Intervals outside `iqr_factor` * IQR will be treated as outliers and

    Returns
    -------
    array of bool, shape=(len(data),)
        An array indicating whether or not each point should be considered
        as an outlier.
    """
    # Get the intervals indexes
    states_start, states_end = get_intervals_idx(states)
    states_duration = states_end - 1 - states_start
    states_duration[0] += 1
    n_intervals = len(states_duration)
    intervals_idx = np.repeat(np.arange(n_intervals), states_duration)
    # Consolidate the results in a dataframe
    df = pd.DataFrame(dict(
        data=data,
        states=states,
        intervals_idx=intervals_idx,
        intervals_start=np.repeat(states_start, states_duration),
        intervals_end=np.repeat(states_end, states_duration),
    ))

    # Compute the mean data value for each interval
    intervals_means = df.groupby(by=["states", "intervals_idx"]).agg(
        {"data": "mean",
         "intervals_start": "first",
         "intervals_end": "last"}
    )

    # For each state, compute the boundaries for the IQR criterion
    states_iqr = intervals_means.reset_index(drop=False).groupby(by="states").agg(
        {"data": [percentile(25), np.median, percentile(75)]}
    ).droplevel(0, axis=1)
    states_iqr["iqr_low"] = (1 - iqr_factor) * states_iqr["median"] + iqr_factor * states_iqr.percentile_25  # noqa E501
    states_iqr["iqr_high"] = (1 - iqr_factor) * states_iqr["median"] + iqr_factor * states_iqr.percentile_75  # noqa 501
    intervals_means = intervals_means.join(states_iqr[["iqr_low", "iqr_high"]])

    # Detect intervals whose average value is outside the IQR interval
    intervals_means["is_outlier"] = np.logical_or(
        intervals_means.data < intervals_means.iqr_low,
        intervals_means.data > intervals_means.iqr_high
    )
    interval_is_outlier = intervals_means.sort_index(level=1).is_outlier.values
    return np.repeat(interval_is_outlier, states_duration)


def fit_hmm(data, ignore_data=None, n_states=2, detect_outliers=True,
            n_points_fit=10000, iqr_factor=3., **kwargs):
    """Fit a hmm on the input data.

    The data is modelled as being emitted by normal distributions whose
    parameters depend on the hidden state.

    For now simply does a random selection of the state and print the kwargs.

    Parameters
    ----------
    data : array, shape=(n_samples,)
        The input data

    ignore_data : array, shape=(n_samples,) or None
        Array of booleans indicating timestamps that will be ignored for the
        fit. If None is provided, all points are used for the fit.

    n_states : int, default=2
        The number of hidden states for the HMM

    detect_outliers : bool, default=True
        If True, after fitting the hmm, for each state, find intervals that
        will be treated as outliers and rerun the pipeline ignoring them.
        The quartiles of the averages of the intervals is computed, we then
        use the inter-quartile-range (IQR) to exclude potential outliers.
        Intervals outside `iqr_factor` * IQR will be treated as outliers and
        ignored for a second run of the pipeline.

    n_points_fit : int, default=10000
        The maximum number of points used for the fit.

    iqr_factor : float, default=3.
        The factor applied to the inter-quartile range for detecting outlier
        intervals.

    **kwargs
        Additional key-word parameters

    Returns
    -------
    states, array of integers, shape=(n_samples,)
        The predicted states. Points without states must be set to -1.

    mu_all, list of floats with length n_states
        The mean parameters for the Gaussian distribution of each state

    sigma_all, list of floats with length n_states
        The std parameters for the Gaussian distribution of each state

    """
    # Use all data if None was provided for ignore_data
    if ignore_data is None:
        ignore_data = np.zeros(len(data), dtype=bool)

    # Print the parameters
    ignore_data_str = f"{ignore_data.sum()} ({100 * ignore_data.sum() / len(data):.1f}%)"
    print(f"Fitting hmm with parameters:")
    params = {"n_states": n_states, "detect_outliers": detect_outliers,
              "n_points_fit": n_points_fit,
              "iqr_factor": iqr_factor,
              "n_points_ignored": ignore_data_str,
              **kwargs}
    for name, value in params.items():
        print(f"    {name} = {value}")

    # Fit the hmm
    print("Fitting HMM")
    data_no_ignore = data[~ignore_data]
    hmm = ssm.HMM(K=n_states, D=1)
    lp = hmm.fit(
        data_no_ignore[:min(len(data_no_ignore), n_points_fit)].reshape(-1, 1),
        verbose=1
    )
    mu_all = hmm.observations.mus.flatten()
    sigma_all = hmm.observations.Sigmas.flatten()
    states_no_ignore = hmm.most_likely_states(data_no_ignore.reshape(-1, 1))
    states = np.full(len(data), -1, dtype=int)
    states[~ignore_data] = states_no_ignore

    # Run the outlier detection pipeline if needed
    if detect_outliers:
        print("Running outlier detection pipeline")
        is_outlier = detect_outlier_intervals(
            data_no_ignore, states_no_ignore, iqr_factor=iqr_factor
        )
        if np.any(is_outlier):
            print("Found outliers, refitting the model.")
            ignore_data[~ignore_data] = is_outlier
            # Refit the model without outlier detection
            return fit_hmm(data, ignore_data=ignore_data, n_states=n_states,
                           detect_outliers=False, n_points_fit=n_points_fit,
                           iqr_factor=iqr_factor, **kwargs)
    return states, mu_all, sigma_all


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
    def from_parameters(cls, mu_all, sigma_all=None, **kwargs):
        "Instanciate the object from the gaussians' parameters."
        # Make sure that the input lengths match
        kwargs.update({"n_states": len(mu_all)})
        # Create the object and set the parameters of the distributions
        self = cls(**kwargs)
        if sigma_all is None:
            sigma_all = np.zeros(len(mu_all))
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
            ignore_data = np.zeros(len(data), dtype=bool)

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
