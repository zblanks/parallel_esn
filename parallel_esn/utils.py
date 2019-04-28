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
        Time series data, which can be 1D or 2D. If 2D, first index
        corresponds to the time, and the second index corresponds to
        a feature in the feature vector. I.e. feature vectors are
        rows stacked vertically in time.
    windowsize : int
        The number of data points in time to be used as input (equal to
        expected number of outputs)
    stride : int
        Number of points to shift in time to generate a subsequent input
        window/output window of data.
    Returns
    -------
    batchU : np.ndarray
        Partitioned samples.
        Dimensions are (num_samples, num_features, windowsize)
    batchY : np.ndarray
        Partitioned targets.
        Dimensions are (num_samples, num_features, windowsize)

    """
    ts_dim = len(timeseries.shape)
    if ts_dim > 2 or ts_dim < 1:
        raise ValueError(("Improper shape passed to chunk_data, "
                          "timeseries must be one or two dimensional. (Received "
                          "timeseries with dimension = {}".format(ts_dim)))
    elif ts_dim == 2:  # feature vector
        feature_len = timeseries.shape[1]
    else:  # single feature
        feature_len = 1
        # promote to 2D array
        timeseries = timeseries.reshape(-1, 1)

    length = timeseries.shape[0]
    # Get transposed view for later array assignment convenience
    timeseriesT = timeseries.T
    if length < 2*windowsize:
        raise ValueError(("Window size too large for timeseries. "
                          "Provided timeseries has {} times, "
                          "and provided windowsize needs at least "
                          "2*windowsize = {} time points."
                          .format(length, 2*windowsize)))
    num_chunks = (length - 2*windowsize)//stride + 1
    batchU = np.zeros((num_chunks, feature_len, windowsize))
    batchY = np.zeros((num_chunks, feature_len, windowsize))
    for i in range(num_chunks):
        start = stride*i
        end = start + windowsize
        batchU[i, :, :] = timeseriesT[:, start:end]
        start = stride*i + windowsize
        end = start + windowsize
        batchY[i, :, :] = timeseriesT[:, start:end]
    return batchU, batchY
