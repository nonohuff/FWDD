import numpy as np
import pytest

from fwdd.filter_function import filter_function_approx, filter_function_finite


def test_filter_function_approx():
    # Test numerical output shape
    omega = np.linspace(0.1, 10.0, 100)
    out = filter_function_approx(omega, 8, 2.0)
    assert out.shape == omega.shape
    assert np.all(out >= 0)

def test_filter_function_finite():
    omega = np.linspace(0.1, 10.0, 100)
    # T must be >= N * tau_p
    with pytest.raises(ValueError):
        filter_function_finite(omega, 8, 1.0, 0.2) # T = 1.0, N * tau_p = 8 * 0.2 = 1.6 > 1.0

    out = filter_function_finite(omega, 8, 2.0, 0.024, method='numba')
    assert out.shape == omega.shape
    assert np.all(out >= 0)

    out_arr = filter_function_finite(omega, 8, 2.0, 0.024, method='numpy_array')
    assert np.allclose(out, out_arr)

    out_py = filter_function_finite(omega, 8, 2.0, 0.024, method='numpy')
    assert np.allclose(out, out_py)
