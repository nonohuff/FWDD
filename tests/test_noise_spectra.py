import numpy as np

from fwdd.noise_spectra import (
    noise_spectrum_1f,
    noise_spectrum_combination,
    noise_spectrum_double_power_law,
    noise_spectrum_lor,
    noise_spectrum_white,
)


def test_noise_spectrum_1f():
    omega = np.array([1.0, 2.0, 3.0])
    out = noise_spectrum_1f(omega, A=2.0, alpha=1.0)
    expected = 2.0 / omega
    assert np.allclose(out, expected)

def test_noise_spectrum_lor():
    omega = np.array([1.0, 2.0, 3.0])
    out = noise_spectrum_lor(omega, omega_0=2.0, gamma=1.0, A=1.5)
    assert out.shape == omega.shape
    assert np.all(out > 0)

def test_noise_spectrum_white():
    omega = np.array([1.0, 2.0, 3.0])
    out = noise_spectrum_white(omega, C=3.14)
    assert np.allclose(out, np.array([3.14, 3.14, 3.14]))

def test_noise_spectrum_double_power_law():
    omega = np.array([1.0, 2.0, 3.0])
    out = noise_spectrum_double_power_law(omega, A=1.0, alpha=1.0, beta=2.0, gamma=1.5)
    assert out.shape == omega.shape
    assert np.all(out > 0)

def test_noise_spectrum_combination():
    omega = np.array([1.0, 2.0, 3.0])
    f_params = {"A": [1.0], "alpha": [1.0]}
    lor_params = {}
    white_params = {"C": [0.5]}
    double_power_law_params = {}

    out = noise_spectrum_combination(
        omega,
        f_params=f_params,
        lor_params=lor_params,
        white_params=white_params,
        double_power_law_params=double_power_law_params
    )
    expected = (1.0 / omega) + 0.5
    assert np.allclose(out, expected)

def test_noise_spectrum_combination_all():
    omega = np.array([1.0, 2.0, 3.0])
    f_params = {"A": [1.0], "alpha": [1.0]}
    lor_params = {"omega_0": [2.0], "gamma": [1.0], "A": [1.5]}
    white_params = {"C": [0.5]}
    double_power_law_params = {"A": [1.0], "alpha": [1.0], "beta": [2.0], "gamma": [1.5]}

    out = noise_spectrum_combination(
        omega,
        f_params=f_params,
        lor_params=lor_params,
        white_params=white_params,
        double_power_law_params=double_power_law_params
    )
    assert out.shape == omega.shape
    assert np.all(out > 0)
