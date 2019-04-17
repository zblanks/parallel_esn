import pytest
import numpy as np
from ..esn import ESN
from ..utils import compute_spectral_radius


def test_initialize_hidden_layer():
    # Check that error is raised when distribution is not uniform or normal
    esn = ESN(input_dim=1, hidden_dim=10, output_dim=1, spectral_radius=0.9,
              k=2, weight_distn='binomial')
    with pytest.raises(ValueError):
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
