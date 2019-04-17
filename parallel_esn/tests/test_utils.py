import numpy as np
import pytest
from ..utils import compute_spectral_radius, create_rng


def test_compute_spectral_radius():
    # Test that a non-square matrix yields an error
    rng = np.random.RandomState(17)
    X = rng.rand(5, 3)
    with pytest.raises(AssertionError):
        compute_spectral_radius(X)

    # A matrix with zeros should have a spectral radius of zero
    X = np.zeros((5, 5))
    rho = compute_spectral_radius(X)
    assert rho == 0.0

    # A matrix with the form: X = [[9, -1, 2], [-2, 8, 4], [1, 1, 8]] has
    # a spectral radius of 10
    X = np.array([[9, -1, 2], [-2, 8, 4], [1, 1, 8]])
    rho = compute_spectral_radius(X)
    assert rho == 10.0


def test_create_rng():
    # If None is passed make sure that it gives a RNG and that it yields the
    # same array
    rng0 = np.random.RandomState(17)
    bytes0 = rng0.bytes(1)
    rng1 = create_rng(None)
    bytes1 = rng1.bytes(1)
    assert bytes0 == bytes1

    # Check if an integer is passed that it yields the same value
    rng1 = create_rng(17)
    bytes1 = rng1.bytes(1)
    assert bytes0 == bytes1

    # Check if RNG is passed that it yields the correct value
    rng1 = create_rng(np.random.RandomState(17))
    bytes1 = rng1.bytes(1)
    assert bytes0 == bytes1
