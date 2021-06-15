"""Functions for fitting an Hidden markov model on data.

Author: Romain Fayat, May 2021
"""
import numpy as np
from functools import wraps


def check_fitted(f):
    "Check that the object is fitted before running the decorated method."

    @wraps(f)
    def decorated(self, *args, **kwargs):
        if not self._fitted:
            raise ValueError("Object was not fitted.")
        return f(self, *args, **kwargs)

    return decorated


def fit_hmm(data, n_states=2, **kwargs):
    """Fit a hmm on the input data.

    The data is modelled as being emitted by normal distributions whose
    parameters depend on the hidden state.

    For now simply does a random selection of the state and print the kwargs.

    Parameters
    ----------
    data : array, shape=(n_samples,)
        The input data

    n_states : int, default=2
        The number of hidden states for the HMM

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
    # Dummy print of the kwargs for prototyping
    print(f"Fitting hmm on {len(data)} samples\nParameters:")
    for name, value in kwargs.items():
        print(f"    {name}={value}")

    # Random allocation of the output states
    states = np.random.choice(n_states, len(data))

    # Compute the mean and std from each states' data
    mu_all = [np.mean(data[states == s]) for s in range(n_states)]
    sigma_all = [np.std(data[states == s]) for s in range(n_states)]
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
