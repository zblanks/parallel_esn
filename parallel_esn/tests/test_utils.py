import numpy as np
import pytest
from ..utils import compute_spectral_radius, create_rng, chunk_data


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


def test_chunk_data_1d():
    # Test chunking 1d data
    timeseries = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # Test when data chunks evenly
    chunkU1, chunkY1 = chunk_data(timeseries, 2, 4)
    ansU1 = np.array([[[0, 1]], [[4, 5]]])
    ansY1 = np.array([[[2, 3]], [[6, 7]]])
    np.testing.assert_array_equal(chunkU1, ansU1)
    np.testing.assert_array_equal(chunkY1, ansY1)

    # Test when data does not chunk evenly
    chunkU2, chunkY2 = chunk_data(timeseries, 2, 3)
    ansU2 = np.array([[[0, 1]], [[3, 4]]])
    ansY2 = np.array([[[2, 3]], [[5, 6]]])
    np.testing.assert_array_equal(chunkU2, ansU2)
    np.testing.assert_array_equal(chunkY2, ansY2)


def test_chunk_data_2d():
    # Test chunking 2d data
    timeseries = np.array([[0, 0],
                           [1, -1],
                           [2, -2],
                           [3, -3],
                           [4, -4],
                           [5, -5],
                           [6, -6],
                           [7, -7]])

    # Test when data chunks evenly
    chunkU1, chunkY1 = chunk_data(timeseries, 2, 4)
    ansU1 = np.array([[[0, 1], [0, -1]], [[4, 5], [-4, -5]]])
    ansY1 = np.array([[[2, 3], [-2, -3]], [[6, 7], [-6, -7]]])
    np.testing.assert_array_equal(chunkU1, ansU1)
    np.testing.assert_array_equal(chunkY1, ansY1)

    # Test when data does not chunk evenly
    chunkU2, chunkY2 = chunk_data(timeseries, 2, 3)
    ansU2 = np.array([[[0, 1], [0, -1]], [[3, 4], [-3, -4]]])
    ansY2 = np.array([[[2, 3], [-2, -3]], [[5, 6], [-5, -6]]])
    np.testing.assert_array_equal(chunkU2, ansU2)
    np.testing.assert_array_equal(chunkY2, ansY2)
