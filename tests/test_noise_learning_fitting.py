import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint
from fwdd.noise_learning_fitting import (
    huber_loss,
    reconstruct_args,
    create_coherence_parameter_constraints,
    loss_function,
    combined_loss_function,
    _global_loss_function,
    create_parameter_constraints,
    fit_coherence_decay,
    fit_coherence_decay_combined,
    fit_noise_spectrum,
)
from fwdd.noise_spectra import noise_spectrum_combination

def test_huber_loss():
    residuals = np.array([0.1, 0.2, -0.05])
    loss = huber_loss(residuals, delta=0.1)
    assert isinstance(loss, float)
    assert loss > 0

def test_reconstruct_args():
    params = [1.5, 0.8, 10.0, 50.0]
    param_structure = [{"A": 1, "alpha": 1}, {"beta": 1, "gamma": 1}]
    reconstructed = reconstruct_args(params, param_structure)
    assert len(reconstructed) == 2
    assert reconstructed[0] == {"A": [1.5], "alpha": [0.8]}
    assert reconstructed[1] == {"beta": [10.0], "gamma": [50.0]}

def test_create_coherence_parameter_constraints():
    param_structure = [{}, {}, {}, {"A": 1, "alpha": 1, "beta": 1, "gamma": 1}]
    constraints = create_coherence_parameter_constraints(param_structure, N1f=0, Nlor=0, NC=0, Ndpl=1)
    assert len(constraints) == 1
    assert isinstance(constraints[0], NonlinearConstraint)

def test_loss_function():
    # Simple evaluation
    params = [1.5, 0.8]
    C_t_observed = np.array([0.8, 0.6])
    times = np.array([1.0, 2.0])
    param_structure = [{"A": 1, "alpha": 1}, {}, {}, {}]
    fixed_kwargs = {"N": 8, "tau_p": 0.024, "integration_method": "trapezoid"}
    
    loss = loss_function(
        params,
        C_t_observed,
        times,
        noise_spectrum_combination,
        param_structure,
        fixed_kwargs,
        delta=True, # use delta for speed
        loss_type="mse"
    )
    assert isinstance(loss, float)

def test_combined_loss_function():
    params = [1.5, 0.8]
    C_t_observed_dict = {8: np.array([0.8, 0.6])}
    times_dict = {8: np.array([1.0, 2.0])}
    param_structure = [{"A": 1, "alpha": 1}, {}, {}, {}]
    fixed_kwargs_dict = {8: {"N": 8, "tau_p": 0.024, "integration_method": "trapezoid"}}
    
    loss = combined_loss_function(
        params,
        C_t_observed_dict,
        times_dict,
        noise_spectrum_combination,
        param_structure,
        fixed_kwargs_dict,
        n_values=[8],
        delta=True
    )
    assert isinstance(loss, float)

def test_fit_coherence_decay():
    C_t_observed = np.array([0.9, 0.7])
    times = np.array([1.0, 2.0])
    initial_args = [{"A": [1.0], "alpha": [1.0]}, {}, {}, {}]
    fixed_kwargs = {"N": 8, "tau_p": 0.024, "integration_method": "trapezoid"}
    bounds = [(0.1, 10.0), (0.1, 5.0)]
    
    opt_args, errors, result = fit_coherence_decay(
        C_t_observed,
        times,
        noise_spectrum_combination,
        initial_args,
        fixed_kwargs,
        bounds=bounds,
        method="L-BFGS-B",
        delta=True,
        use_constraints=False
    )
    assert len(opt_args) == 4
    assert "A" in opt_args[0]

def test_fit_coherence_decay_combined():
    C_t_observed_dict = {8: np.array([0.9, 0.7])}
    times_dict = {8: np.array([1.0, 2.0])}
    initial_args = [{"A": [1.0], "alpha": [1.0]}, {}, {}, {}]
    fixed_kwargs_dict = {8: {"N": 8, "tau_p": 0.024, "integration_method": "trapezoid"}}
    bounds = [(0.1, 10.0), (0.1, 5.0)]
    
    opt_args, errors, result = fit_coherence_decay_combined(
        C_t_observed_dict,
        times_dict,
        noise_spectrum_combination,
        initial_args,
        fixed_kwargs_dict,
        n_values=[8],
        bounds=bounds,
        method="L-BFGS-B",
        delta=True,
        use_constraints=False
    )
    assert len(opt_args) == 4
    assert "A" in opt_args[0]

def test_global_loss_function():
    params = [1.5, 0.8]
    freq_points = np.array([1.0, 2.0, 3.0])
    S_w_observed = np.array([1.5, 0.75, 0.5])
    param_structure = [{"A": 1, "alpha": 1}]
    
    loss = _global_loss_function(
        params,
        freq_points,
        S_w_observed,
        param_structure,
        loss_type="mse"
    )
    assert isinstance(loss, float)

def test_fit_noise_spectrum():
    freq_points = np.array([1.0, 2.0, 3.0])
    S_w_observed = np.array([10.0, 5.0, 3.33])
    
    opt_args, result = fit_noise_spectrum(
        freq_points,
        S_w_observed,
        N1f=1,
        Nlor=0,
        NC=0,
        Ndpl=0,
        method="L-BFGS-B",
        use_constraints=False
    )
    assert len(opt_args) == 4
    assert "A" in opt_args[0]
