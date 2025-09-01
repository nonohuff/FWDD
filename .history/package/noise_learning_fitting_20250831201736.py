
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, differential_evolution
from scipy.special import huber

from sklearn.metrics import mean_squared_error

from skopt import gp_minimize
from skopt.space import Real

from tqdm.notebook import tqdm, trange
from tqdm.contrib import tenumerate

import multiprocessing

from package import filter_function as ff
from package import coherence_profile as cp

# Perform curve fitting
def func_to_fit(times, noise_profile, *args, delta=False, **kwargs):
    '''
    Function to fit the coherence profile to the observed noise profile.
    Inputs:
    - times: Array of time values
    - noise_profile: Function representing the noise profile
    - args: Additional arguments for the noise profile function
    - delta: Boolean indicating if delta function should be used
    - kwargs: Additional keyword arguments
    Outputs:
    - C_t: Coherence profile at the given times
    '''
    n = kwargs.pop('N', None) 
    tau_p = kwargs.pop('tau_p', None)
    integration_method = kwargs.pop('integration_method', None)

    if delta:
        # For delta function case, we need to evaluate the noise profile at the appropriate frequencies
        omega_values = n * np.pi / np.array(times)
        noise_spectrum_values = noise_profile(omega_values, *args)
        C_t = cp.coherence_decay_profile_delta(np.array(times), noise_spectrum_values)
    else:
        num_cpus = multiprocessing.cpu_count()
        if int(num_cpus) > 1:
            C_t, _, _, _ = cp.parallel_coherence_decay(times, n, tau_p, integration_method, noise_profile, *args, initial_n_jobs=int(num_cpus), max_memory_per_worker=10**4, **kwargs)
        else:
            C_t, _, _, _ = zip(*[cp.coherence_decay_profile_finite_peaks_with_widths(t, n, tau_p, integration_method, noise_profile, *args, **kwargs) for t in times])

    # Check for NaN or infinite values and handle them
    C_t = np.array(C_t)
    if np.any(np.isnan(C_t)) or np.any(np.isinf(C_t)):
        # Return a large value array to penalize bad parameter combinations
        C_t = np.full_like(C_t, 1e10)
    
    return C_t


def huber_loss(residuals, delta=None):
    """
    Huber loss function using scipy.special.huber
    
    Parameters:
    -----------
    residuals : array-like
        Prediction errors (y_true - y_pred)
    delta : float, optional
        Threshold parameter. If None, automatically set to 1.345 * MAD
    
    Returns:
    --------
    float
        Mean Huber loss
    """
    residuals = np.array(residuals)
    
    if delta is None:
        # Automatic delta selection using MAD (Median Absolute Deviation)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        delta = 1.345 * mad * 1.4826  # 1.4826 is consistency factor for normal distribution
        if delta == 0:  # Handle case where all residuals are identical
            delta = 0.01  # Small default value
    
    # scipy.special.huber returns (delta, r) where r is the Huber loss value
    # for each residual scaled by delta
    huber_values = huber(delta, residuals)
    
    # The huber function returns the loss for each point
    # We need to return the mean
    return np.mean(huber_values)

def loss_function(params, C_t_observed, times, noise_profile, param_structure, 
                  fixed_kwargs, delta=False, loss_type='mse', noise_level=None):
    """
    Calculate loss between predicted and observed C_t.
    
    Parameters:
    -----------
    loss_type : str
        'mse' for mean squared error, 'huber' for Huber loss
    noise_level : float, optional
        Known noise level for setting Huber delta parameter
    """
    # Convert params to numpy array if it's a list
    if isinstance(params, list):
        params = np.array(params)
    
    # Reconstruct args from flattened params
    reconstructed_args = []
    param_index = 0
    
    for param_dict in param_structure:
        new_dict = {}
        for key, length in param_dict.items():
            if length > 0:
                param_slice = params[param_index:param_index+length]
                new_dict[key] = param_slice.tolist() if hasattr(param_slice, 'tolist') else list(param_slice)
                param_index += length
        reconstructed_args.append(new_dict)
    
    # Make a prediction
    C_t_pred = func_to_fit(times, noise_profile, *reconstructed_args, delta=delta, **fixed_kwargs)
    
    # Convert to numpy arrays
    C_t_pred = np.array(C_t_pred)
    C_t_observed = np.array(C_t_observed)
    
    # Handle NaN or infinite values
    if np.any(np.isnan(C_t_pred)) or np.any(np.isinf(C_t_pred)):
        return 1e10
    
    # Calculate residuals
    residuals = C_t_observed - C_t_pred
    
    # Calculate and return the loss
    if loss_type == 'mse':
        return mean_squared_error(C_t_observed, C_t_pred)
    elif loss_type == 'huber':
        # For your noise level, set delta based on the expected noise
        if noise_level is not None:
            delta_huber = 1.5 * noise_level  # Can tune this multiplier
        else:
            delta_huber = None  # Will use MAD-based estimation
        return huber_loss(residuals, delta_huber)

def get_parameter_errors(result, C_t_observed, times, noise_profile, param_structure, fixed_kwargs, delta=False):
    # Get the Hessian matrix
    hess_inv = result.hess_inv
    
    # Calculate residuals at the optimal point
    residuals = np.array(C_t_observed) - np.array(func_to_fit(times, noise_profile, 
                                                            *reconstruct_args(result.x, param_structure), 
                                                            delta=delta, **fixed_kwargs))
    
    # Calculate MSE (mean squared error)
    mse = np.sum(residuals**2) / (len(residuals) - len(result.x))
    
    # Covariance matrix = mse * hess_inv
    if isinstance(hess_inv, np.ndarray):
        covariance = mse * hess_inv
    else:
        # For L-BFGS-B, hess_inv might be a LinearOperator
        covariance = mse * hess_inv.todense()
    
    # Standard errors are the square root of the diagonal elements
    parameter_errors = np.sqrt(np.diag(covariance))
    
    return parameter_errors

# Helper function to reconstruct args from flattened params
def reconstruct_args(params, param_structure):
    # Convert params to numpy array if it's a list
    if isinstance(params, list):
        params = np.array(params)
        
    reconstructed_args = []
    param_index = 0
    
    for param_dict in param_structure:
        new_dict = {}
        for key, length in param_dict.items():
            if length > 0:
                param_slice = params[param_index:param_index+length]
                new_dict[key] = param_slice.tolist() if hasattr(param_slice, 'tolist') else list(param_slice)
                param_index += length
        reconstructed_args.append(new_dict)
    
    return reconstructed_args

def fit_coherence_decay(C_t_observed, times, noise_profile, initial_args, fixed_kwargs, bounds=None, method='L-BFGS-B', delta=False, loss_type="mse", noise_level=None):
    """
    Fit the coherence decay function to observed data.
    
    Parameters:
    -----------
    C_t_observed : array-like
        Experimentally observed coherence decay
    times : array-like
        Time points for coherence profile
    noise_profile : function
        Noise spectrum function
    initial_args : list of dicts
        Initial guesses for the parameters to optimize
    fixed_kwargs : dict
        Fixed parameters that are not optimized
    bounds : list of tuples, optional
        Bounds for each parameter (lower, upper)
    method : str, optional
        Optimization method to use
    delta : bool, optional
        Whether to use delta function approach (default: False)
    
    Returns:
    --------
    tuple
        (optimized_args, optimized_errors, optimization_result)
    """
    # Create a structure that describes the parameter dictionaries
    param_structure = []
    flattened_initial_params = []
    
    for arg_dict in initial_args:
        structure_dict = {}
        for key, value in arg_dict.items():
            structure_dict[key] = len(value)
            flattened_initial_params.extend(value)
        param_structure.append(structure_dict)

 # Perform the optimization
    if method == 'gp_minimize':
        # Use Gaussian Process optimization for expensive functions
        
        if bounds is None:
            raise ValueError("Bounds are required for Gaussian Process optimization")
        
        # Convert bounds to skopt dimensions
        dimensions = []
        for i, bound in enumerate(bounds):
            if bound is None or bound[0] is None or bound[1] is None:
                Warning("All bounds must be finite (min, max) tuples for differential evolution. Setting bounds to finite numbers.")
                # Replace any infinite bounds with large finite numbers
            lower = -1e6 if (bound[0] == -np.inf or bound[0] == None)  else bound[0]
            upper = 1e6 if (bound[1] == -np.inf or bound[1] == None)  else bound[1]
            dimensions.append(Real(lower, upper, name=f'param_{i}'))
        
        # Wrapper function for skopt (expects parameter list)
        def skopt_objective(x):
            return loss_function(x, C_t_observed, times, noise_profile, param_structure, fixed_kwargs, delta, loss_type, noise_level)
        
        result = gp_minimize(
            func=skopt_objective,
            dimensions=dimensions,
            n_calls=50,  # Total evaluation budget for expensive functions
            # n_initial_points=max(10, 2*len(dimensions)),  # Initial random sampling
            popsize=10,
            n_initial_points=10,  # Initial random sampling
            mutation=(0.5, 1.5),  # Adaptive mutation range
            recombination=0.7,  # Adaptive recombination rate
            acq_func='gp_hedge',  # Adaptive acquisition function
            noise=1e-10,  # Small noise for numerical stability
            n_restarts_optimizer=5,  # Robustness for acquisition optimization
            verbose = True,  # Show progress
            init='latinhypercube',
            updating='deferred',
            n_jobs=-1 if delta else 1,  # Use all available CPUs
        )

    elif method == 'diff_ev':
        # Use differential evolution for expensive functions
        
        if bounds is None:
            raise ValueError("Bounds are required for differential evolution")
        
        # Convert bounds format if needed - ensure all bounds are finite
        clean_bounds = []
        for bound in bounds:
            if bound is None or bound[0] is None or bound[1] is None:
                Warning("All bounds must be finite (min, max) tuples for differential evolution. Setting bounds to finite numbers.")
                # Replace any infinite bounds with large finite numbers
            lower = -1e6 if (bound[0] == -np.inf or bound[0] == None)  else bound[0]
            upper = 1e6 if (bound[1] == -np.inf or bound[1] == None)  else bound[1]
            clean_bounds.append((lower, upper))
        
        result = differential_evolution(
            func=loss_function,
            bounds=clean_bounds,
            args=(C_t_observed, times, noise_profile, param_structure, fixed_kwargs, delta, loss_type, noise_level),
            updating = "deferred" if delta else "immediate",
            workers = -1 if delta else 1,
            strategy ='best1bin',  # Strategy for differential evolution
            maxiter=5000,  # Limit generations for expensive functions
            init = "halton",
            popsize=20,   # Small population
            polish=True, # Skip final local polish to save evaluations
            tol = 1e-6,  # Tolerance for convergence}
            disp=True    # Show progress
        )
        
    elif method == 'L-BFGS-B':
        try:
            result = minimize(
                loss_function,
                np.array(flattened_initial_params),
                args=(C_t_observed, times, noise_profile, param_structure, fixed_kwargs, delta, loss_type, noise_level),
                bounds=bounds,
                method=method,
                options={'ftol': 1e-6, 'gtol': 1e-5, 'maxls': 100}
            )
        except Exception as e:
            result = minimize(
                loss_function,
                np.array(flattened_initial_params),
                args=(C_t_observed, times, noise_profile, param_structure, fixed_kwargs, delta, loss_type, noise_level),
                bounds=bounds,
                method=method,
            )  
    else:
        result = minimize(
            loss_function,
            np.array(flattened_initial_params),
            args=(C_t_observed, times, noise_profile, param_structure, fixed_kwargs, delta, loss_type, noise_level),
            bounds=bounds,
            method=method,
        )  

    # Convert result.x to numpy array if it's a list (from gp_minimize)
    if isinstance(result.x, list):
        result_x = np.array(result.x)
    else:
        result_x = result.x
    
    # Reconstruct the optimized args with errors
    optimized_args = []
    optimized_errors = []
    param_index = 0
    
    for param_dict in param_structure:
        new_dict = {}
        error_dict = {}
        for key, length in param_dict.items():
            if length > 0:
                param_slice = result_x[param_index:param_index+length]
                new_dict[key] = param_slice.tolist() if hasattr(param_slice, 'tolist') else list(param_slice)
                # error_dict[key] = parameter_errors[param_index:param_index+length].tolist()
                param_index += length
        optimized_args.append(new_dict)
        # optimized_errors.append(error_dict)
    
    return optimized_args, optimized_errors, result