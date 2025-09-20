import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.special import huber

from sklearn.metrics import mean_squared_error

from skopt import gp_minimize
from skopt.space import Real

import multiprocessing

from package import coherence_profile as cp
from package.noise_spectra import noise_spectrum_combination

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


def create_coherence_parameter_constraints(param_structure, N1f, Nlor, NC, Ndpl):
    """
    Create constraints for coherence decay optimization problem.
    For double power law noise: beta > alpha.
    
    Parameters:
    -----------
    param_structure : list
        Structure of parameter dictionaries
    N1f : int
        Number of 1/f noise parameters
    Nlor : int
        Number of Lorentzian noise parameters
    NC : int
        Number of white noise parameters
    Ndpl : int
        Number of double power law noise parameters
    
    Returns:
    --------
    constraints : list
        List of constraint dictionaries for scipy.optimize   
    """
    
    constraints = []
    
    if Ndpl > 0:
        # Find the indices for double power law parameters
        param_index = 0
        
        # Skip 1/f noise parameters
        if N1f > 0:
            param_index += 2 * N1f  # A and alpha
            
        # Skip Lorentzian parameters  
        if Nlor > 0:
            param_index += 3 * Nlor  # A, omega_0, gamma
            
        # Skip white noise parameters
        if NC > 0:
            param_index += NC  # C
            
        # Now we're at double power law parameters
        # Order is: A, alpha, beta, gamma (repeated for each component)
        for i in range(Ndpl):
            # For each double power law component
            alpha_index = param_index + Ndpl + i  # alpha is second in the flattened array
            beta_index = param_index + 2*Ndpl + i  # beta is third
            
            def constraint_func(params, alpha_idx=alpha_index, beta_idx=beta_index):
                """beta > alpha constraint with small tolerance"""
                return params[beta_idx] - params[alpha_idx] - 1e-6
            
            constraint = NonlinearConstraint(constraint_func, 0, np.inf)
            constraints.append(constraint)
    
    return constraints

def fit_coherence_decay(C_t_observed, times, noise_profile, initial_args, fixed_kwargs, 
                        bounds=None, method='diff_ev', delta=False, loss_type="mse", 
                        noise_level=None, population=10, iterations=200, use_constraints=True):
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
    loss_type : str, optional
        Type of loss function to use (default: 'mse')
    noise_level : float, optional
        Estimated noise level for Huber loss (default: None)
    use_constraints : bool, optional
        Whether to apply beta > alpha constraint for double power law noise

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

    # Extract parameter counts from fixed_kwargs or infer from initial_args
    N1f = fixed_kwargs.get('N1f', len(initial_args[0].get('A', [])) if len(initial_args) > 0 else 0)
    Nlor = fixed_kwargs.get('Nlor', len(initial_args[1].get('A', [])) if len(initial_args) > 1 else 0)
    NC = fixed_kwargs.get('NC', len(initial_args[2].get('C', [])) if len(initial_args) > 2 else 0)
    Ndpl = fixed_kwargs.get('Ndpl', len(initial_args[3].get('A', [])) if len(initial_args) > 3 else 0)

    # Create constraints if requested
    constraints = []
    if use_constraints and Ndpl > 0:
        constraints = create_coherence_parameter_constraints(param_structure, N1f, Nlor, NC, Ndpl)
        print(f"Created {len(constraints)} constraints for beta > alpha")

     # Perform the optimization
    if method == 'gp_minimize':
        # Use Gaussian Process optimization for expensive functions. This is a global optimization method.
        
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
            n_calls=iterations,  # Total evaluation budget for expensive functions
            popsize=population,
            # n_initial_points = max(10, 2*len(dimensions)),  # Initial random sampling (may have to reduce if computationally expensive.)
            n_initial_points = 2*len(dimensions) if delta else 10, # small number of initial points since C(t) is computationally expensive
            mutation=(0.5, 1.5),  # Adaptive mutation range
            recombination=0.7,  # Adaptive recombination rate
            acq_func='gp_hedge',  # Adaptive acquisition function
            noise=1e-10,  # Small noise for numerical stability
            n_restarts_optimizer=5,  # Robustness for acquisition optimization
            verbose = True,  # Show progress
            init='latinhypercube',
            updating='deferred',
            n_jobs=-1 if delta else 1,  # parallelize if using the delta function approximation, otherwise, we are already parallelizing in the function that computes C(t)
        )

    elif method == 'diff_ev':
        # Use differential evolution for expensive functions. This is also a global optimization method, and it is gradient-free.
        
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
            bounds = clean_bounds,
            args=(C_t_observed, times, noise_profile, param_structure, fixed_kwargs, delta, loss_type, noise_level),
            constraints=constraints if constraints else (),  # Add constraints here
            updating = "deferred" if delta else "immediate",
            workers = -1 if delta else 1, # parallelize if using the delta function approximation, otherwise, we are already parallelizing in the function that computes C(t)
            strategy ='best1bin',  # Strategy for differential evolution
            maxiter = iterations,  # Limit generations for expensive functions
            init = "halton",
            popsize = population,   # Small population
            polish = True, # final local optimization polish with 'L-BFGS-B'
            tol = 1e-6,  # Tolerance for convergence
            disp = True    # Show progress
        )
        
    elif method == 'L-BFGS-B':
        # Use L-BFGS-B for local optimization (local optimizer)
        try:
            result = minimize(
                loss_function,
                np.array(flattened_initial_params),
                args=(C_t_observed, times, noise_profile, param_structure, fixed_kwargs, delta, loss_type, noise_level),
                bounds=bounds,
                constraints=constraints if constraints else (),  # Add constraints here
                method=method,
                options={'ftol': 1e-6, 'gtol': 1e-5, 'maxls': 100}
            )
        except Exception as e:
            result = minimize(
                loss_function,
                np.array(flattened_initial_params),
                args=(C_t_observed, times, noise_profile, param_structure, fixed_kwargs, delta, loss_type, noise_level),
                bounds=bounds,
                constraints=constraints if constraints else (),  # Add constraints here
                method=method,
            )  
    else:
        # if the optimization method is not one of the three above, we pass it as an arguemtn to scipy minimize, if you'd like to try other optimizers.
        result = minimize(
            loss_function,
            np.array(flattened_initial_params),
            args=(C_t_observed, times, noise_profile, param_structure, fixed_kwargs, delta, loss_type, noise_level),
            bounds=bounds,
            constraints=constraints if constraints else (),  # Add constraints here
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
        # error_dict = {}
        for key, length in param_dict.items():
            if length > 0:
                param_slice = result_x[param_index:param_index+length]
                new_dict[key] = param_slice.tolist() if hasattr(param_slice, 'tolist') else list(param_slice)
                # error_dict[key] = parameter_errors[param_index:param_index+length].tolist()
                param_index += length
        optimized_args.append(new_dict)
        # optimized_errors.append(error_dict)
    
    # Verify constraints are satisfied (if they were applied)
    if use_constraints and Ndpl > 0:
        print("\nConstraint verification:")
        for i, args in enumerate(optimized_args):
            if 'beta' in args and 'alpha' in args:
                for j in range(len(args['beta'])):
                    beta_val = args['beta'][j]
                    alpha_val = args['alpha'][j]
                    constraint_satisfied = beta_val > alpha_val
                    print(f"Component {j}: beta={beta_val:.6f}, alpha={alpha_val:.6f}, "
                          f"beta > alpha: {constraint_satisfied}")
    
    return optimized_args, optimized_errors, result

def combined_loss_function(params, C_t_observed_dict, times_dict, noise_profile, param_structure, 
                          fixed_kwargs_dict, n_values, delta=False, loss_type='huber', noise_level=None):
    """
    Calculate combined loss across multiple n values.
    
    Parameters:
    -----------
    params : array-like
        Flattened parameter array
    C_t_observed_dict : dict
        Dictionary with n values as keys and observed C_t arrays as values
    times_dict : dict
        Dictionary with n values as keys and time arrays as values
    noise_profile : function
        Noise spectrum function
    param_structure : list
        Parameter structure for reconstruction
    fixed_kwargs_dict : dict
        Dictionary with n values as keys and fixed parameters as values
    n_values : list
        List of n values to combine
    delta : bool
        Whether to use delta approximation
    loss_type : str
        Type of loss function ('mse' or 'huber')
    noise_level : float
        Noise level for Huber loss
    
    Returns:
    --------
    float
        Combined loss across all n values
    """
    total_loss = 0.0
    
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
    
    # Calculate loss for each n value
    for n in n_values:
        try:
            # Get the specific parameters for this n value
            fixed_kwargs = fixed_kwargs_dict[n]
            times = times_dict[n]
            C_t_observed = C_t_observed_dict[n]
            
            # Make prediction for this n value
            C_t_pred = func_to_fit(times, noise_profile, *reconstructed_args, delta=delta, **fixed_kwargs)
            
            # Convert to numpy arrays
            C_t_pred = np.array(C_t_pred)
            C_t_observed = np.array(C_t_observed)
            
            # Handle NaN or infinite values
            if np.any(np.isnan(C_t_pred)) or np.any(np.isinf(C_t_pred)):
                return 1e10
            
            # Calculate residuals
            residuals = C_t_observed - C_t_pred
            
            # Calculate loss for this n value
            if loss_type == 'mse':
                n_loss = mean_squared_error(C_t_observed, C_t_pred)
            elif loss_type == 'huber':
                if noise_level is not None:
                    delta_huber = 1.5 * noise_level
                else:
                    delta_huber = None
                n_loss = huber_loss(residuals, delta_huber)
            
            total_loss += n_loss
            
        except Exception as e:
            print(f"Error calculating loss for n={n}: {e}")
            return 1e10
    
    return total_loss

def fit_coherence_decay_combined(C_t_observed_dict, times_dict, noise_profile, initial_args, 
                                fixed_kwargs_dict, n_values, bounds=None, method='L-BFGS-B', 
                                delta=False, loss_type="mse", noise_level=None, population=10, 
                                iterations=200, use_constraints=True):
    """
    Fit the coherence decay function to observed data across multiple n values.
    
    Parameters:
    -----------
    C_t_observed_dict : dict
        Dictionary with n values as keys and observed C_t arrays as values
    times_dict : dict
        Dictionary with n values as keys and time arrays as values
    noise_profile : function
        Noise spectrum function
    initial_args : list of dicts
        Initial guesses for the parameters to optimize
    fixed_kwargs_dict : dict
        Dictionary with n values as keys and fixed parameters as values
    n_values : list
        List of n values to combine
    bounds : list of tuples, optional
        Bounds for each parameter (lower, upper)
    method : str, optional
        Optimization method to use
    delta : bool, optional
        Whether to use delta function approach (default: False)
    loss_type : str, optional
        Type of loss function ('mse' or 'huber')
    noise_level : float, optional
        Noise level for Huber loss
    use_constraints : bool, optional
        Whether to apply beta > alpha constraint for double power law noise
    
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

    # Extract parameter counts from the first fixed_kwargs entry or infer from initial_args
    first_kwargs = fixed_kwargs_dict[n_values[0]]
    N1f = first_kwargs.get('N1f', len(initial_args[0].get('A', [])) if len(initial_args) > 0 else 0)
    Nlor = first_kwargs.get('Nlor', len(initial_args[1].get('A', [])) if len(initial_args) > 1 else 0)
    NC = first_kwargs.get('NC', len(initial_args[2].get('C', [])) if len(initial_args) > 2 else 0)
    Ndpl = first_kwargs.get('Ndpl', len(initial_args[3].get('A', [])) if len(initial_args) > 3 else 0)

    # Create constraints if requested
    constraints = []
    if use_constraints and Ndpl > 0:
        constraints = create_coherence_parameter_constraints(param_structure, N1f, Nlor, NC, Ndpl)
        print(f"Created {len(constraints)} constraints for beta > alpha")

    # Perform the optimization
    if method == 'gp_minimize':
        if bounds is None:
            raise ValueError("Bounds are required for Gaussian Process optimization")
        
        # Convert bounds to skopt dimensions
        dimensions = []
        for i, bound in enumerate(bounds):
            if bound is None or bound[0] is None or bound[1] is None:
                Warning("All bounds must be finite (min, max) tuples for differential evolution. Setting bounds to finite numbers.")
            lower = -1e6 if (bound[0] == -np.inf or bound[0] == None) else bound[0]
            upper = 1e6 if (bound[1] == -np.inf or bound[1] == None) else bound[1]
            dimensions.append(Real(lower, upper, name=f'param_{i}'))
        
        # Wrapper function for skopt
        def skopt_objective(x):
            return combined_loss_function(x, C_t_observed_dict, times_dict, noise_profile, 
                                        param_structure, fixed_kwargs_dict, n_values, 
                                        delta, loss_type, noise_level)
        
        result = gp_minimize(
            func=skopt_objective,
            dimensions=dimensions,
            n_calls=iterations,
            popsize=population,
            n_initial_points=10,
            mutation=(0.5, 1.5),
            recombination=0.7,
            acq_func='gp_hedge',
            noise=1e-10,
            n_restarts_optimizer=5,
            verbose=True,
            init='latinhypercube',
            updating='deferred',
            n_jobs=-1 if delta else 1
        )

    elif method == 'diff_ev':
        if bounds is None:
            raise ValueError("Bounds are required for differential evolution")
        
        # Convert bounds format if needed
        clean_bounds = []
        for bound in bounds:
            if bound is None or bound[0] is None or bound[1] is None:
                Warning("All bounds must be finite (min, max) tuples for differential evolution. Setting bounds to finite numbers.")
            lower = -1e6 if (bound[0] == -np.inf or bound[0] == None) else bound[0]
            upper = 1e6 if (bound[1] == -np.inf or bound[1] == None) else bound[1]
            clean_bounds.append((lower, upper))
        

        result = differential_evolution(
            func=combined_loss_function,
            bounds=clean_bounds,
            args=(C_t_observed_dict, times_dict, noise_profile, param_structure, 
                  fixed_kwargs_dict, n_values, delta, loss_type, noise_level),
            constraints=constraints if constraints else (),  # Add constraints here
            strategy='best1bin',
            updating = "deferred" if delta else "immediate",
            init = "halton",
            workers= -1 if delta else 1,
            maxiter=iterations,
            popsize=population,
            polish=True,
            tol=1e-6,
            disp=True
        )
        
    elif method == 'L-BFGS-B':
        try:
            result = minimize(
                combined_loss_function,
                np.array(flattened_initial_params),
                args=(C_t_observed_dict, times_dict, noise_profile, param_structure, 
                      fixed_kwargs_dict, n_values, delta, loss_type, noise_level),
                bounds=bounds,
                constraints=constraints if constraints else (),  # Add constraints here
                method=method,
                options={'ftol': 1e-6, 'gtol': 1e-5, 'maxls': 100}
            )
        except Exception as e:
            result = minimize(
                combined_loss_function,
                np.array(flattened_initial_params),
                args=(C_t_observed_dict, times_dict, noise_profile, param_structure, 
                      fixed_kwargs_dict, n_values, delta, loss_type, noise_level),
                bounds=bounds,
                constraints=constraints if constraints else (),  # Add constraints here
                method=method,
            )
    else:
        result = minimize(
            combined_loss_function,
            np.array(flattened_initial_params),
            args=(C_t_observed_dict, times_dict, noise_profile, param_structure, 
                  fixed_kwargs_dict, n_values, delta, loss_type, noise_level),
            bounds=bounds,
            constraints=constraints if constraints else (),  # Add constraints here
            method=method,
        )

    # Convert result.x to numpy array if it's a list
    if isinstance(result.x, list):
        result_x = np.array(result.x)
    else:
        result_x = result.x
    
    # Reconstruct the optimized args
    optimized_args = []
    optimized_errors = []  # Placeholder for now
    param_index = 0
    
    for param_dict in param_structure:
        new_dict = {}
        for key, length in param_dict.items():
            if length > 0:
                param_slice = result_x[param_index:param_index+length]
                new_dict[key] = param_slice.tolist() if hasattr(param_slice, 'tolist') else list(param_slice)
                param_index += length
        optimized_args.append(new_dict)
    
    # Verify constraints are satisfied (if they were applied)
    if use_constraints and Ndpl > 0:
        print("\nConstraint verification:")
        for i, args in enumerate(optimized_args):
            if 'beta' in args and 'alpha' in args:
                for j in range(len(args['beta'])):
                    beta_val = args['beta'][j]
                    alpha_val = args['alpha'][j]
                    constraint_satisfied = beta_val > alpha_val
                    print(f"Component {j}: beta={beta_val:.6f}, alpha={alpha_val:.6f}, "
                          f"beta > alpha: {constraint_satisfied}")
    
    return optimized_args, optimized_errors, result


###########################################################################################################################
# For fitting S(ω) data to a model
###########################################################################################################################

# Global loss function for multiprocessing compatibility
def _global_loss_function(params, freq_points, S_w_observed, param_structure, loss_type='mse'):
    """Calculate loss between predicted and observed noise spectrum.
    Inputs:
    - params: Flattened list of parameters to optimize
    - freq_points: Frequency points (omega values)
    - S_w_observed: Observed noise spectrum values
    - param_structure: Structure to reconstruct parameter dictionaries
    - loss_type: Type of loss function ('mse', 'log_mse', 'rel_mse')
    Outputs:
    - loss: Calculated loss value
    """
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
    
    # Calculate predicted noise spectrum
    try:
        S_w_pred = noise_spectrum_combination(freq_points, *reconstructed_args)
        
        # Handle NaN or infinite values
        if np.any(np.isnan(S_w_pred)) or np.any(np.isinf(S_w_pred)):
            return 1e10
        
        # Calculate loss
        if loss_type == 'mse':
            return mean_squared_error(S_w_observed, S_w_pred)
        elif loss_type == 'log_mse':
            # Add other loss functions here if needed
            return mean_squared_error(np.log10(S_w_observed), np.log10(S_w_pred))
            # return np.mean((np.log(S_w_observed) - np.log(S_w_pred))**2)
        elif loss_type == 'rel_mse':
            return mean_squared_error(S_w_observed/S_w_observed, S_w_pred/S_w_observed)
            
    except Exception as e:
        print(f"Error in loss calculation: {e}")
        return 1e10

def create_parameter_constraints(param_structure, N1f, Nlor, NC, Ndpl):
    """
    Create constraints for the optimization problem.
    For double power law noise: beta > alpha.
    Inputs:
    - param_structure: Structure of parameter dictionaries
    - N1f: Number of 1/f noise parameters
    - Nlor: Number of Lorentzian noise parameters
    - NC: Number of white noise parameters
    - Ndpl: Number of double power law noise parameters
    Outputs:
    - constraints: List of constraint dictionaries for scipy.optimize   
    """
    
    constraints = []
    
    if Ndpl > 0:
        # Find the indices for double power law parameters
        param_index = 0
        
        # Skip 1/f noise parameters
        if N1f > 0:
            param_index += 2 * N1f  # A and alpha
            
        # Skip Lorentzian parameters  
        if Nlor > 0:
            param_index += 3 * Nlor  # beta, gamma, A
            
        # Skip white noise parameters
        if NC > 0:
            param_index += NC  # C
            
        # Now we're at double power law parameters
        # Order is: A, alpha, beta, gamma (repeated for each component)
        for i in range(Ndpl):
            # For each double power law component
            alpha_index = param_index + Ndpl + i  # alpha is second in the flattened array
            beta_index = param_index + 2*Ndpl + i  # beta is third
            
            def constraint_func(params, alpha_idx=alpha_index, beta_idx=beta_index):
                """beta > alpha constraint with small tolerance"""
                return params[beta_idx] - params[alpha_idx] - 1e-6
            
            constraint = NonlinearConstraint(constraint_func, 0, np.inf)
            constraints.append(constraint)
    
    return constraints

def fit_noise_spectrum(freq_points, S_w_observed, N1f=1, Nlor=0, NC=0, Ndpl=0, 
                      bounds=None, method='L-BFGS-B', loss_type='mse', use_constraints=True, population=5000, iterations=1000):
    """
    Fit noise_spectrum_combination to observed noise spectrum data.
    
    Parameters:
    -----------
    freq_points : array-like
        Frequency points (omega values)
    S_w_observed : array-like
        Observed noise spectrum values
    N1f : int
        Number of 1/f noise parameters
    Nlor : int
        Number of Lorentzian noise parameters
    NC : int
        Number of white noise parameters
    Ndpl : int
        Number of double power law noise parameters
    bounds : list of tuples, optional
        Bounds for each parameter (lower, upper)
    method : str, optional
        Optimization method to use
    loss_type : str, optional
        Loss function type. One of ["mse", "rel_mse", or "log_mse"] ('mse' for mean squared error)
    use_constraints : bool, optional
        Whether to apply beta > alpha constraint for double power law noise
    
    Returns:
    --------
    tuple
        (optimized_args, optimization_result)
    """
    
    # Create initial parameter structure
    initial_args = []
    param_structure = []
    flattened_initial_params = []
    
    # 1/f noise parameters
    if N1f > 0:
        f_params = {'A': [10.0] * N1f, 'alpha': [1.0] * N1f}
        initial_args.append(f_params)
        param_structure.append({'A': N1f, 'alpha': N1f})
        flattened_initial_params.extend(f_params['A'] + f_params['alpha'])
    else:
        initial_args.append({})
        param_structure.append({})
    
    # Lorentzian parameters
    if Nlor > 0:
        lor_params = {'beta': [1.0] * Nlor, 'gamma': [1.0] * Nlor, 'A': [1.0] * Nlor}
        initial_args.append(lor_params)
        param_structure.append({'beta': Nlor, 'gamma': Nlor, 'A': Nlor})
        flattened_initial_params.extend(lor_params['beta'] + lor_params['gamma'] + lor_params['A'])
    else:
        initial_args.append({})
        param_structure.append({})
    
    # White noise parameters
    if NC > 0:
        white_params = {'C': [1.0] * NC}
        initial_args.append(white_params)
        param_structure.append({'C': NC})
        flattened_initial_params.extend(white_params['C'])
    else:
        initial_args.append({})
        param_structure.append({})
    
    # double power law parameters
    if Ndpl > 0:
        # Modified initial values to satisfy beta > alpha constraint
        cf_params = {'A': [1.0] * Ndpl, 'alpha': [1.0] * Ndpl, 
                    'beta': [2.0] * Ndpl, 'gamma': [50] * Ndpl}  # beta > alpha initially
        initial_args.append(cf_params)
        param_structure.append({'A': Ndpl, 'alpha': Ndpl, 'beta': Ndpl, 'gamma': Ndpl})
        flattened_initial_params.extend(cf_params['A'] + cf_params['alpha'] + cf_params['beta'] + cf_params['gamma'])
    else:
        initial_args.append({})
        param_structure.append({})
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = []
        # 1/f noise bounds
        bounds.extend([(0, 1000)] * N1f)  # A bounds
        bounds.extend([(0, 5)] * N1f)     # alpha bounds
        # Lorentzian bounds
        bounds.extend([(1e-4, 1e8)] * Nlor)   # beta bounds
        bounds.extend([(1e-4, 1e6)] * Nlor)   # gamma bounds
        bounds.extend([(0, 1000)] * Nlor)     # A bounds
        # White noise bounds
        bounds.extend([(0, 100)] * NC)        # C bounds
        # Double Power Law bounds - Modified to allow beta > alpha
        bounds.extend([(1e-10, 10**3)] * Ndpl)  # A bounds
        bounds.extend([(1e-6, 10)] * Ndpl)      # alpha bounds (min > 0 for constraint)
        bounds.extend([(1e-5, 20)] * Ndpl)   # beta bounds (min > alpha_min)
        bounds.extend([(1e-10, 100)] * Ndpl)  # gamma bounds
    
    # Create constraints if requested
    constraints = []
    if use_constraints and Ndpl > 0:
        constraints = create_parameter_constraints(param_structure, N1f, Nlor, NC, Ndpl)
        print(f"Created {len(constraints)} constraints for beta > alpha")
    
    # Perform optimization
    if method == 'diff_ev':
        # Use differential evolution for global optimization
        result = differential_evolution(
            _global_loss_function,
            bounds,
            args=(freq_points, S_w_observed, param_structure, loss_type),
            constraints=constraints if constraints else (),  # Add constraints here
            strategy='best1bin',
            updating="deferred",  # or "immediate"
            workers=-1,  # Can use multiple workers now
            init="halton",
            maxiter=iterations,
            popsize=population,
            polish=True,
            tol=1e-6,
            disp=True
        )
    else:
        # Use local optimization
        result = minimize(
            _global_loss_function,
            np.array(flattened_initial_params),
            args=(freq_points, S_w_observed, param_structure, loss_type),
            bounds=bounds,
            constraints=constraints if constraints else (),  # Add constraints here
            method=method,
            options={'ftol': 1e-6, 'gtol': 1e-5}
        )
    
    # Reconstruct the optimized args
    optimized_args = []
    param_index = 0
    
    result_x = np.array(result.x) if isinstance(result.x, list) else result.x
    
    for param_dict in param_structure:
        new_dict = {}
        for key, length in param_dict.items():
            if length > 0:
                param_slice = result_x[param_index:param_index+length]
                new_dict[key] = param_slice.tolist() if hasattr(param_slice, 'tolist') else list(param_slice)
                param_index += length
        optimized_args.append(new_dict)
    
    # Verify constraints are satisfied (if they were applied)
    if use_constraints and Ndpl > 0:
        print("\nConstraint verification:")
        for i, args in enumerate(optimized_args):
            if 'beta' in args and 'alpha' in args:
                for j in range(len(args['beta'])):
                    beta_val = args['beta'][j]
                    alpha_val = args['alpha'][j]
                    constraint_satisfied = beta_val > alpha_val
                    print(f"Component {j}: beta={beta_val:.6f}, alpha={alpha_val:.6f}, "
                          f"beta > alpha: {constraint_satisfied}")
    
    return optimized_args, result