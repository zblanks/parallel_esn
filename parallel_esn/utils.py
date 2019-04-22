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


def chunk_data(timeseries, windowsize, stride):
    """
    Partitions time series data

    Parameters
    ----------
    timeseries : np.ndarray
        Time series data
    windowsize : int
        The number of data points to be used as input (equal to expected number
        of outputs)
    stride : int
        Number of points to shift in time to generate a subsequent input
        window/output window of data.
    Returns
    -------
    batchU : np.ndarray
        Partitioned samples
    batchY : np.ndarray
        Partitioned targets

    """
    length = timeseries.shape[0]
    num_chunks = (length-(2*windowsize - 1))//stride
    batchU = np.zeros((num_chunks, 1, windowsize))
    batchY = np.zeros((num_chunks, 1, windowsize))
    for i in range(num_chunks):
        start = stride*i
        end = start + windowsize
        batchU[i, 0, :] = timeseries[start:end]
        start = stride*i + windowsize
        end = start + windowsize
        batchY[i, 0, :] = timeseries[start:end]
    return batchU, batchY
