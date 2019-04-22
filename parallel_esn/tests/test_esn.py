import pytest
import numpy as np
from ..esn import ESN
from ..utils import compute_spectral_radius


def test_initialize_hidden_layer():
    # Check that error is raised when distribution is not uniform or normal
    with pytest.raises(ValueError):
        esn = ESN(input_dim=1, hidden_dim=10, output_dim=1, spectral_radius=0.9, k=2, weight_distn='binomial')
        esn._initialize_hidden_layer()

    # Check if matrix has desired spectral radius
    esn = ESN(input_dim=1, hidden_dim=10, output_dim=1, k=2,
              spectral_radius=0.9)
    W = esn._initialize_hidden_layer()
    spectral_radius = compute_spectral_radius(W)
    np.testing.assert_approx_equal(spectral_radius, 0.9)


def test_initialize_input_layer():
    # Check that matrix has the correct shape
    esn = ESN(input_dim=1, hidden_dim=10, k=2, output_dim=1)
    desired_shape = (10, 2)
    W_in = esn._initialize_input_layer()
    assert W_in.shape == desired_shape


def test_train_validate():
    # Run a simple test of training on sinusoidal data
    t = np.linspace(0, 100, 1001)
    data = np.sin(2*np.pi*t)
    # Create a validation the same way, with a phase shift
    val_t = np.linspace(0, 30, 301)
    val_data = np.sin(2*np.pi*val_t + np.sqrt(2))

    def chunk_data(timeseries, windowsize, stride):
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

    windowsize = 100
    trainU, trainY = chunk_data(data, windowsize, 40)
    valU, valY = chunk_data(val_data, windowsize, 40)
    # Create a new ESN
    esn = ESN(1, windowsize, 1, 3)
    losses = esn.train(trainU, trainY, verbose=0)
    # Check that the right number of losses are returned
    assert len(losses) == trainU.shape[0]
    # Check that the algorithm makes progress
    assert losses[0] >= losses[-1]

    val_loss = esn.validate(valU, valY)
    # Check that the validation loss is nonnegative
    assert val_loss >= 0.
