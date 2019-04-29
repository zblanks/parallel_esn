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


def chunk_data(timeseries, windowsize, stride, predict_cols=[]):
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
    predict_cols : array_like of int, optional
        List of column indices of timeseries to return in batchY. Should
        correspond to the features you want to predict. Implied order
        of features in the list is maintained in returned batchY.

    Returns
    -------
    batchU : np.ndarray
        Partitioned samples.
        Dimensions are (num_samples, num_features, windowsize)
    batchY : np.ndarray
        Partitioned targets.
        Dimensions are (num_samples, len(predict_cols), windowsize)

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

    # Number of prediction timeseries
    pred_len = len(predict_cols)
    if pred_len == 0:  # Then predict all columns
        predict_cols = range(feature_len)
        pred_len = len(predict_cols)

    # Ensure that columns to predict are in passed timeseries
    for col in predict_cols:
        if col > feature_len-1 or col < 0:
            raise ValueError("Column index {} in predict_cols does not "
                             "correspond to a column in provided "
                             "timeseries, which has {} columns"
                             .format(col, feature_len))
    # How many time slices are in the timeseries
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
    batchY = np.zeros((num_chunks, pred_len, windowsize))
    for i in range(num_chunks):
        start = stride*i
        end = start + windowsize
        batchU[i, :, :] = timeseriesT[:, start:end]
        start = stride*i + windowsize
        end = start + windowsize
        for j, col in enumerate(predict_cols):
            batchY[i, j, :] = timeseriesT[col, start:end]
    return batchU, batchY


def standardize_traindata(timeseries):
    """
    Standardizes train data by column. Each column is assumed to correspond
    to a feature, and the returned features are shifted and rescaled by

    y' = (y - mu)/sigma

    where y is a feature, mu is the mean of the feature (over time) and sigma
    is the standard deviation of the feature (over time). An array containing
    mu and an array containing sigma for each feature is returned for
    use in transforming validation and test data. These arrays can be
    passed alongside validation/test data to the scale_data function
    which will then scale the data in the same way that the training
    data was scaled.

    In principle, the time series passed should only contain training
    data to ensure that no information about the validation and test
    data is used in training the network. Validation and test data
    should then be scaled with mu_arr and sigma_arr using the function
    scale_data.

    Parameters
    ----------
    timeseries : np.ndarray
        Time series data, which can be 1D or 2D. If 2D, first index
        corresponds to the time, and the second index corresponds to
        a feature in the feature vector. I.e. feature vectors are
        rows stacked vertically in time. Intended to be training data.

    Returns
    -------
    std_timeseries : np.ndarray
        Time series data, with dimensions matching input time series.
        Every feature is standardized based on the mean and standard
        deviation calculated from its time series.
    mu_arr : np.ndarray
        Means of each feature, in the same order as the order of the
        columns in the provided time series
    sigma_arr : np.ndarray
        Standard deviations of each feature, in the same order as the
        order of the columns in the provided time series.

    """
    if len(timeseries.shape) == 1:
        mu_arr = np.zeros(1)
        sigma_arr = np.zeros(1)
        mu = np.mean(timeseries)
        sigma = np.std(timeseries)
        if sigma == 0.:
            sigma = 1.
        std_timeseries = (timeseries - mu) / sigma
        mu_arr[0] = mu
        sigma_arr[0] = sigma
    elif len(timeseries.shape) == 2:
        mu_arr = np.zeros(timeseries.shape[1])
        sigma_arr = np.zeros(timeseries.shape[1])
        std_timeseries = np.zeros_like(timeseries)
        for j in range(timeseries.shape[1]):
            mu = np.mean(timeseries[:, j])
            sigma = np.std(timeseries[:, j])
            if sigma == 0.:
                sigma = 1.
            std_timeseries[:, j] = (timeseries[:, j] - mu) / sigma
            mu_arr[j] = mu
            sigma_arr[j] = sigma
    else:
        raise ValueError("Timeseries does not have appropriate shape; must be "
                         "1D or 2D.")
    return std_timeseries, mu_arr, sigma_arr


def scale_data(timeseries, mu_arr, sigma_arr):
    """
    Scales time series data by the standardization set by
    training data. The scaling is performed by the equation

    y' = (y - mu)/sigma

    where y is a feature, mu is the mean of the feature (over time) in the
    training data and sigma is the standard deviation of the feature (over time)
    in the training data. An array containing mu and sigma for each feature
    is returned for use in transforming validation and test data.

    Intended to be used only if training data was standardized with the
    function standardize_traindata.

    Parameters
    ----------
    timeseries : np.ndarray
        Time series data, which can be 1D or 2D. If 2D, first index
        corresponds to the time, and the second index corresponds to
        a feature in the feature vector. I.e. feature vectors are
        rows stacked vertically in time.
    mu_arr : np.ndarray
        Means of each feature computed from training data, in the
        same order as the order of the columns in the provided time series
    sigma_arr : np.ndarray
        Standard deviations of each feature from training data, in
        the same order as the order of the columns in the provided time
        series.

    Returns
    -------
    std_timeseries : np.ndarray
        Time series data, with dimensions matching input time series.
        Every feature is standardized based on the mean and standard
        deviation calculated from its corresponding training data.

    """
    if len(timeseries.shape) == 1:
        std_timeseries = (timeseries - mu_arr[0]) / sigma_arr[0]
    elif len(timeseries.shape) == 2:
        std_timeseries = np.zeros_like(timeseries)
        for j in range(timeseries.shape[1]):
            std_timeseries[:, j] = (timeseries[:, j] - mu_arr[j]) / sigma_arr[j]
    else:
        raise ValueError("Timeseries does not have appropriate shape; must be "
                         "1D or 2D.")
    return std_timeseries


def unscale_data(scaled_pred, mu_arr, sigma_arr, predict_cols=[]):
    """
    Scales time series data by the standardization set by
    training data. The scaling is performed by the equation

    y' = (y - mu)/sigma

    where y is a feature, mu is the mean of the feature (over time) in the
    training data and sigma is the standard deviation of the feature (over time)
    in the training data. An array containing mu and sigma for each feature
    is returned for use in transforming validation and test data.

    Intended to be used if training data was standardized with function
    standardize_traindata, and if validation/test data was scaled by the
    function scale_data.

    Parameters
    ----------
    scaled_pred : np.ndarray
        Scaled prediction data arising from running the echo state network
        on standardized input data.
    mu_arr : np.ndarray
        Means of each feature computed from training data, in the
        same order as the order of the columns in the provided time series
    sigma_arr : np.ndarray
        Standard deviations of each feature from training data, in
        the same order as the order of the columns in the provided time
        series.
    predict_cols : array-like, optional
        A list of indices corresponding to predicted features in scaled_pred,
        with the value they would have if they indexed all the features in
        the *input* data. If chunk_data was called with a specified list
        of indices in optional argument predict_cols, the same list should
        be passed to unscale_data here.

    Returns
    -------
    pred : np.ndarray
        Prediction data transformed back to the scale and units of the original
        timeseries, prior to standardization.

    """
    if len(mu_arr) != len(sigma_arr):
        raise ValueError("mu_arr and sigma_arr do not have the same number "
                         "of elements")
    if len(scaled_pred.shape) == 1:
        pred = (scaled_pred * sigma_arr[0]) + mu_arr[0]
    elif len(scaled_pred.shape) == 2:
        pred_len = len(predict_cols)
        if pred_len == 0:  # Then all features were predicted.
            predict_cols = range(scaled_pred.shape[1])
            pred_len = len(predict_cols)
        else:
            # Ensure that column indices corresponding to predicted features
            # are in mu_arr and sigma_arr
            for col in predict_cols:
                if col > len(mu_arr)-1 or col < 0:
                    raise ValueError("Column index {} in predict_cols does not "
                                     "correspond to a column in provided "
                                     "mu_arr and sigma_arr, which have {} columns"
                                     .format(col, len(mu_arr)))
        pred = np.zeros_like(scaled_pred)
        for j, col in enumerate(predict_cols):
            pred[:, j] = (scaled_pred[:, j] * sigma_arr[col]) + mu_arr[col]
    else:
        raise ValueError("scaled_pred does not have appropriate shape; must be "
                         "1D or 2D.")
    return pred
