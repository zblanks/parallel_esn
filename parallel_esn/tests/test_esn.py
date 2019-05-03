import pytest
import numpy as np
from ..esn import ESN
from ..utils import compute_spectral_radius, chunk_data, to_forecast_form


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


@pytest.fixture(scope="module", autouse=True)
def generate_data_network():
    # Create data for the following tests. Only runs
    # once in the testing for efficiency.
    # Run a simple test of training on sinusoidal data
    t = np.linspace(0, 10, 101)
    data = np.sin(2*np.pi*t)
    # Create a validation the same way, with a phase shift
    val_t = np.linspace(0, 3, 31)
    val_data = np.sin(2*np.pi*val_t + np.sqrt(2))
    hidden = 10
    trainU, trainY = chunk_data(data, hidden, 4)
    valU, valY = chunk_data(val_data, hidden, 4)

    esn = ESN(1, hidden, 1, 3)
    losses = esn.train(trainU, trainY, verbose=0, compute_loss_freq=1, warmup=0)

    return t, val_t, trainU, trainY, valU, valY, esn, losses


def test_train_validate(generate_data_network):
    # Test training and validation

    # Get data and trained network from generate_data_network
    t, val_t, trainU, trainY, valU, valY, esn, losses = generate_data_network

    # Check that the right number of losses are returned
    assert len(losses) == trainU.shape[0]
    # Check that the algorithm makes progress
    assert losses[0] >= losses[-1]

    val_loss = esn.validate(valU, valY, warmup=0)
    # Check that the validation loss is better than the first loss in training
    assert 0 <= val_loss <= losses[0]


def test_score_with_X(generate_data_network):
    # Run a confirmation that scoring/predicting with X is
    # equivalent to doing so with U.

    # Get data and trained network from generate_data_network
    t, val_t, trainU, trainY, valU, valY, esn, losses = generate_data_network

    for s in range(trainU.shape[0]):
        esn.clear_state()
        X = esn._compute_X(trainU[s, :, :])
        esn.clear_state()
        prediction_from_U = esn.predict(trainU[s, :, :])
        prediction_from_X = esn.predict_with_X(X)
        np.testing.assert_allclose(prediction_from_U, prediction_from_X)

        esn.clear_state()
        score_from_U = esn.score(trainU[s], trainY[s])
        esn.clear_state()
        score_from_X = esn.score_with_X(X, trainY[s])
        np.testing.assert_allclose(score_from_U, score_from_X)


def test_cython_equivalence():
    # This is to test if the cython _compute_X behaves the same
    # way as the numpy version.
    # Create a training set of sinusoidal data
    t = np.linspace(0, 10, 101)
    data = np.sin(2*np.pi*t)
    # Create a validation the same way, with a phase shift
    val_t = np.linspace(0, 3, 31)
    val_data = np.sin(2*np.pi*val_t + np.sqrt(2))
    hidden = 10
    trainU, trainY = chunk_data(data, hidden, 4)
    valU, valY = chunk_data(val_data, hidden, 4)

    esn = ESN(1, hidden, 1, 3)
    # Cython losses
    losses_cy = esn.train(trainU, trainY, verbose=0, compute_loss_freq=1, warmup=0)
    esn.reset()
    esn.use_cython = False
    losses_np = esn.train(trainU, trainY, verbose=0, compute_loss_freq=1, warmup=0)
    np.testing.assert_allclose(losses_cy, losses_np)


@pytest.fixture(scope="module", autouse=True)
def generate_recursive_network():
    # Create data for the following tests. Only runs
    # once in the testing for efficiency.
    # Run a simple test of training on sinusoidal data
    t = np.linspace(0, 10, 101)
    data = np.sin(2*np.pi*t)
    # Create a validation the same way, with a phase shift
    val_t = np.linspace(0, 3, 31)
    val_data = np.sin(2*np.pi*val_t + np.sqrt(2))
    hidden = 10
    trainU, trainY, _, _ = to_forecast_form(data, -1)
    valU, valY, _, _ = to_forecast_form(val_data, -1)

    esn = ESN(1, hidden, 1, 3)
    esn.train(trainU, trainY, verbose=0, compute_loss_freq=-1, warmup=0)

    return t, val_t, trainU, trainY, valU, valY, esn


def test_recursive_predict(generate_recursive_network):
    t, val_t, trainU, trainY, valU, valY, esn = generate_recursive_network
    print(trainU)
    Yhat = esn.recursive_predict(trainU[0, :10, :], 10)
    assert Yhat.shape[1] == 10


def test_recursive_score(generate_recursive_network):
    t, val_t, trainU, trainY, valU, valY, esn = generate_recursive_network
    error = esn.recursive_score(valU[0, :, :], valY[0, :, :], 20, 10)
    assert error < 0.5
