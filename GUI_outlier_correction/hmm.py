"""Functions for fitting an Hidden markov model on data

Author: Romain Fayat, May 2021
"""

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
    print(f"Fitting hmm on {len(data)} samples\n"
           "Parameters:")
    for name, value in kwargs.items():
        print(f"    {name}={value}")

    # Random allocation of the output states
    states = np.random.choice(n_states, len(data))

    # Compute the mean and std from each states' data
    mu_all = [np.mean(data[states == s]) for s in range(n_states)]
    sigma_all = [np.std(data[states == s]) for s in range(n_states)]
    return states, mu_all, sigma_all
