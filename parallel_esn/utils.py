import numpy as np


def compute_spectral_radius(X):
    """
    Computes the spectral radius of a matrix

    Parameters
    ----------
    X : np.ndarray
        Input matrix

    Returns
    -------
    rho : float
        Spectral radius of `X`

    """
    # To compute the eigen-values of a matrix, it must be square
    assert X.shape[0] == X.shape[1], print('Input matrix must be square')

    eigvals = np.linalg.eigvals(X)
    return np.max(np.abs(eigvals))


def create_rng(random_state):
    """
    Creates a random state object

    Parameters
    ----------
    random_state : int or NoneType or np.random.RandomState
        Input to create RNG

    Returns
    -------
    rng : np.random.RandomState
        Pseudo-random number generator

    """
    if random_state is None:
        return np.random.RandomState(17)
    elif isinstance(random_state, int):
        return np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        raise TypeError("Must pass either a None, integer, or "
                        "np.random.RandomState")
