import pytest
from ..bo import BO


def test_init():
    # Check that error is thrown when we don't pass a tuple for a
    # hyper-parameter
    with pytest.raises(AssertionError):
        BO(k=2)


def test_update_gpr():
    # Check that the data has been update and that the GPR has changed after
    # adding new data
    bo = BO(k=(2, 3))
    X = [100, 2, 1., 0.1, 0.5, 1e-3]
    y = 0.1234
    orig_data_len = len(bo.H)
    bo.update_gpr(X, y)
    new_data_len = len(bo.H)
    assert new_data_len > orig_data_len


def test_build_options():
    # Check that the matrix has 6 columns for the hyper-parameters and
    # has the corresponding number of samples
    bo = BO(k=(2, 10))
    H_space = bo._build_options(num_samples=100)
    assert H_space.shape == (100, 6)


def test_find_best_choice():
    # Check returned value is a dictionary with six values for the
    # hyper-parameters
    bo = BO(k=(2, 10))
    param_vals = bo.find_best_choices()
    assert len(param_vals) == 6
