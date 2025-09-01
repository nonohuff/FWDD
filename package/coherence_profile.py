import os
import psutil
import time

import numpy as np
import numba as nb

import joblib
from joblib import Parallel, delayed, parallel_backend

from scipy.signal import find_peaks
from scipy.integrate import quad, simpson, trapezoid, IntegrationWarning

from package import filter_function as ff 

# Custom Exception Classes
class ParallelExecutionError(Exception):
    """Custom exception for parallel execution failures"""
    pass

class MemoryThresholdError(Exception):
    """Custom exception for memory threshold exceeded"""
    pass

# define the function to compute coherence decay using a delta function approximation
@nb.njit(parallel=False)
def coherence_decay_profile_delta(t,noise_profile):
    """
    Calculate the coherence decay profile based on the provided formula, e**(-χ(t)),
    under the assumption that the filter function is a delta function, (i.e. χ(t)=t*S(ω)/π).
    Inputs:
    noise_profile: a function that implements the noise spectrum, S(omega)
    t: a float of the time variable
    Output:
    e**(-χ(t))
    """
    chi_t = t*noise_profile/np.pi
    return np.exp(-chi_t)

def noise_inversion_delta(t , C_t):
    """
    Calculate the noise, S(ω), based on the provided formula, C(t) = 1 - e**(-χ(t)),
    under the assumption that the filter function is a delta function, (i.e. χ(t)=t*S(ω)/π).
    Inputs:
    t: a float of the time variable
    C_t: a float of the coherence decay profile at time t
    Output:
    S(ω)
    """
    return -np.pi*np.log(C_t)/t

def monitor_memory():
    """Monitor current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

def find_k_largest_peaks(arr, k):
    """
    Find the k largest peaks in a 1D array.
    
    Parameters:
    -----------
    arr : array-like
        Input 1D array to find peaks in
    k : int
        Number of largest peaks to return
    
    Returns:
    --------
    peak_indices : ndarray
        Indices of the k largest peaks
    peak_heights : ndarray
        Heights of the k largest peaks
    """
    # Find all peaks
    peaks, _ = find_peaks(arr)
    
    # Get peak heights
    peak_heights = arr[peaks]
    
    # Sort peaks by height in descending order
    sorted_peak_indices = np.argsort(peak_heights)[::-1]
    
    # Select k largest peaks
    k_largest_peak_indices = peaks[sorted_peak_indices[:k]]
    k_largest_peak_heights = peak_heights[sorted_peak_indices[:k]]
    
    return k_largest_peak_indices, k_largest_peak_heights


##### 
# Note on computing C(t):
# Performing this integral is challenging due to the oscillatory nature of the integrand and the presence of sharp peaks and the log scaling of ω over which the noise can be relevant.
# In order to tackle these challenges, we employ a combination of techniques, including adaptive quadrature and careful ω selection.
# adaptive quadrature integration tends to be slow and inaccurate due to the sharp peaks. So, we reccomned using one of the included Riemann methods "trapezoid" or "simpson".
# We numerically evaluate these integrals by carefully selecting the integration points, especially around the peaks of the integrand, so that we can capture the important features of the integrand without requiring an excessively fine grid.
# We begin with a coarse grid defined by kwargs["omega_range"] and kwargs["omega_resolution"], and add to these points any singularities from the noise, S(ω), and the approximate locations of the first M peaks near ω = (2m+1)n*π/t.
# Then, we compute what the integrand would be and find the largest points in the integrand, and add kwargs["peak_resolution"] points around each of these peaks to better resolve them.
# The integrand is computed one final time, and this array is used to perform the Riemann integration.
#####


### FYI, there are two hyperparameters that you currently can't feed into this function. "k = 10**4" and "M=100", explained below. 
# You can edit this function to make them passable parameters if you wish, or just edit them directly. ###
def coherence_decay_profile_finite_peaks_with_widths(t, N, tau_p, method, noise_profile, *args, narrow_window = True, max_memory_mb=15000, **kwargs):
    """Computes the integral in eq #1 in [Meneses,Wise,Pagliero,et.al. 2022] exactly, considering finite pulse widths, which computes the Coherence decay profile, C(t). 
    FYI, there are two hyperparameters that you currently can't feed into this function. "k = 10**4" and "M=100", explained below. You can edit this function to make them passable parameters if you wish, or just edit them directly.
    Inputs:
    t: a float of the time variable
    N: a float of the number of pulses
    tau_p: a float of the pulse width
    method: a string of the method to use for integration. Options are "quad", "trapezoid", "simpson", and "log_sum_exp".
    noise_profile: a function that implements the noise spectrum, S(ω), or a numpy array of the noise spectrum
    *args: a list of arguments to pass to the noise_profile function
    **kwargs: a dictionary of keyword arguments to pass to the function
    Output:
    e**(-χ(t)): The coherence decay profile, C(t), the omega values used in the integration, the filter function values, and the integrand values.
    omega_values: a numpy array of angular frequencies
    filter_values: a numpy array of the filter function values, evaluated at omega_values
    integrand: a numpy array of the integrand values given by S(ω)*F(ω*t)/(2*π*ω**2), evaluated at omega_values
    """
    if not (isinstance(N, int) and N > 0):
        raise TypeError(f"N must be a positive integer, got {type(N).__name__}")
    
    if kwargs.get('num_peaks_cutoff'):
        if not (isinstance(kwargs['num_peaks_cutoff'], int) and kwargs['num_peaks_cutoff'] >= 0):
            raise TypeError(f"num_peaks_cutoff must be a positive integer, got {type(kwargs['num_peaks_cutoff']).__name__}")
        
    current_memory = monitor_memory()
    if current_memory > max_memory_mb:
        raise MemoryError(f"Memory usage ({current_memory}MB) exceeded threshold")

    # If narrow_window is enabled, adjust the omega range so that the lower bound is set to 0.5*N*np.pi/t. This avoids wasting numerical points
    #  at low frequencies, where the filter function is small. Disable narrow_window if you want full control over the integration window in ω.
    if narrow_window:
        old_range = kwargs.get("omega_range")  # Default range if not provided
        kwargs["omega_range"] = (0.5*N*np.pi/t,old_range[1]) # Set the lower bound of the omega range to 0.5*N*np.pi/tau_p
     
    if callable(noise_profile): # If noise_profile is a function
        try:
            omegas = np.logspace(np.log10(kwargs.get("omega_range")[0]), np.log10(kwargs.get("omega_range")[1]), 10**7) # setting default resolution to 10^7 points. Might be a little overkill
        except KeyError:
            omegas = np.logspace(-4, 8, 10**7) # setting resolution to 10^7 points. Might be a little overkill  

        S_w = noise_profile(omegas, *args)  # Check that the noise_profile function works with the omega_values

        # First, we identify the k largest values. k is arbitrary, and you can increase it if you want more resolution.
        k = 10**4
        # Find the k largest values.
        # k_largest_values = np.partition(S_w, -k)[-k:] # not needed, but included in case you need it for something else.
        k_largest_indices = np.argpartition(S_w, -k)[-k:]

        local_maxima_indices, _ =  find_peaks(S_w)
        local_minima_indices, _ = find_peaks(-S_w)

        points_of_interest = np.unique(np.concatenate((omegas[local_maxima_indices], omegas[local_minima_indices], omegas[k_largest_indices])))
        # print(len(points_of_interest))

        # Find the peaks of the filter function, located at approximately ω = (2m+1)n*π/t for integer m
        base = np.pi * N / t

        M = 100
        omega_peaks = np.array([base * (2*m + 1) for m in range(M)]) # consider the first 100 FF resonance peaks.
        # 100 peaks is an arbitrary choice here. You can include more peaks if you want more resolution.

        # Check for additional FF resonance peaks near the points of interest on the noise
        # additional_filter_peaks = []
        # for point in points_of_interest:
        #     k = round((point/base - 1)/2)
        #     additional_filter_peaks.append(base * (2*k + 1))
        # additional_filter_peaks = np.array(additional_filter_peaks)

        k_values = np.round((points_of_interest/base - 1)/2).astype(int)
        additional_filter_peaks = base * (2*k_values + 1)

        try:
            omega_sweep = np.logspace(np.log10(kwargs.get("omega_range")[0]), np.log10(kwargs.get("omega_range")[1]), kwargs.get("omega_resolution",10**5)) 
        except KeyError:
            omega_sweep = np.logspace(-4, 8, kwargs.get("omega_resolution",10**5))

        breakpoints = np.unique(np.concatenate((omega_peaks, additional_filter_peaks, points_of_interest)))
        omega_values = np.unique(np.concatenate((breakpoints, omega_sweep)))

        filter_values = ff.filter_function_finite(omega_values, N, t, tau_p)

        integrand = np.multiply(filter_values, np.divide(noise_profile(omega_values, *args),(np.power(omega_values,2))))

        # Now that we've computed the integrand once, we find the largest peaks in the integrand, which will contribute most to the integral. 
        largest_peak_indices, _ = find_k_largest_peaks(integrand, kwargs.get('num_peaks_cutoff', len(omega_values)))

        # for each of these peaks, we will add kwargs["peak_resolution"] additional ω points around them for more precise integration.
        if list(largest_peak_indices): # If there are singularities in the filter function within the specified omega range
            additional_omega_values = np.array([])
            for index in largest_peak_indices:
                if kwargs.get("peak_resolution"):
                # Joonhee says that the width of the peaks should scale as O(1/N)
                # Using 1/N here is not percise, for low values of N, it can lead to negative values of omega being added to omega_values (i.e. peak_width/2 > peak), which I 
                # correct for by removing negative values below. A more sophisticated approach would be to use a peak width that scales with N.
                    peak = omega_values[index]
                    peak_width = 1/N # Width in indices

                    # Otherwise, add additional points around the peak, if no resolution is specified, use 10 points.
                    peak_res = kwargs.get("peak_resolution", 10)

                    # You could opt to use a logspace around the peak, but I went with a linear space for now. Have to fix issue where peak_width/2 > peak, which would require a 
                    # more sophisticated approach to setting peak_width.
                    # additional_omega_values.append(omega_values, np.logspace(np.log10(peak-peak*peak_width), np.log10(peak+peak*peak_width), peak_res)) # uncomment if you wish to use logspace
                    additional_omega_values = np.append(additional_omega_values,np.linspace(peak-peak*peak_width, peak+peak*peak_width, peak_res))
                    
            additional_omega_values = additional_omega_values[additional_omega_values > 0] # Remove non-positive values to be safe. Important when N=1 to avoid devide by zero errors.
            additional_filter_values = ff.filter_function_finite(additional_omega_values, N, t, tau_p)
            additional_integrand = np.multiply(additional_filter_values, np.divide(noise_profile(additional_omega_values, *args),(np.power(additional_omega_values,2))))

            # Concatenate and get unique, sorted omega values
            merged_omega_values = np.sort(np.unique(np.concatenate((omega_values, additional_omega_values))))
            # Create merged filter values array
            merged_integrand = np.zeros_like(merged_omega_values, dtype=integrand.dtype)
            merged_filter_values = np.zeros_like(merged_omega_values, dtype=filter_values.dtype)
            
            # Find indices of original omega values in merged array
            original_indices = np.searchsorted(merged_omega_values, omega_values)
            merged_integrand[original_indices] = integrand
            merged_filter_values[original_indices] = filter_values
            
            # Find indices of new omega values in merged array
            new_indices = np.searchsorted(merged_omega_values, additional_omega_values)
            merged_integrand[new_indices] = additional_integrand
            merged_filter_values[new_indices] = additional_filter_values

            omega_values = merged_omega_values
            integrand = merged_integrand
            filter_values = merged_filter_values
        
            # print("Range of Omega Values: ", np.format_float_scientific(np.min(omega_values)), np.format_float_scientific(np.max(omega_values)))
    else: # If the noise_profile the user specified is not a function, (i.e. tt's an array of the noise values at certain frequencies)
        # In this case, we don't get to re-define the frequency values to better evaluate the integral. Instead, we have to take the frequency points the user gives us.
        # The default is to assume the frequency values are taken in logspace, and defined by kwargs["omega_range"] and kwargs["omega_resolution"]. If this is not the case, you will need to specify 
        # omega_values directly below.
        try:
            S_w = np.array(noise_profile)
            if not np.issubdtype(S_w.dtype, np.number):
                raise ValueError("The noise_profile must be a numerical object broadcastable numpy array.")
        except Exception as e:
            raise ValueError("The noise_profile must be a function or a numerical object broadcastable numpy array.") from e
        
        try:
            omega_values = np.logspace(np.log10(kwargs.get("omega_range")[0]), np.log10(kwargs.get("omega_range")[1]), kwargs.get("omega_resolution",len(S_w))) 
        except KeyError:
            omega_values = np.logspace(-4, 8, kwargs.get("omega_resolution",len(S_w)))

        filter_values = ff.filter_function_finite(omega_values, N, t, tau_p)
        integrand = np.multiply(filter_values, np.divide(S_w, (np.power(omega_values,2))))

    if method == "quad":
        if not callable(noise_profile):
            raise ValueError("The noise_profile must be a function when using the quad method. Try using the trapezoid or simpson method instead.")

        # for the scipy adaptive quadrature, we need to provide the integrand as a function of omega
        integrand = lambda omega: (noise_profile(omega, *args) * filter_function_finite(omega, N, t, tau_p)) / np.power(omega,2)
        # Integrate the integrand over the specified omega range using a maximum of "limit" subdivisions in the adaptive alogrithm,
        #  and using the omega_peaks as break points where sigularities occur.

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore', category=IntegrationWarning)
        chi_t = 0
        chi_t_i, _ = quad(integrand, 0, breakpoints[0], 
                    limit=kwargs.get('quad_limit',len(breakpoints)+10), 
                    points=[0,breakpoints[0]],
                    epsabs=kwargs.get('epsabs', 1.49e-8),
                    epsrel=kwargs.get('epsrel', 1.49e-8))
        chi_t += chi_t_i
        for i in range(len(breakpoints)-1):
            chi_t_i, _ = quad(integrand, breakpoints[i], breakpoints[i+1], 
                                limit=kwargs.get('quad_limit',len(breakpoints)+10), 
                                points=[breakpoints[i],breakpoints[i+1]],
                                epsabs=kwargs.get('epsabs', 1.49e-8),
                                epsrel=kwargs.get('epsrel', 1.49e-8))
            chi_t += chi_t_i
            
        chi_t = chi_t/(np.pi)

        return np.exp(-chi_t), omega_values, filter_values, integrand(omega_values)/(np.pi)  # No omega_values, filter_values, or integrand for quad method
    else:     

        # integrand = np.multiply(filter_values, np.divide(noise_profile(omega_values, *args),(np.power(omega_values,2))))

        # I'm using the "trapezoid" method because it's error has better scaling with the size of the step size of x values, which 
        # is large since we are using log spacing for points.
        if method == "simpson":
            chi_t = simpson(integrand, x=omega_values)
        elif method == "trapezoid":
            chi_t = trapezoid(integrand, x=omega_values)
        else:
            raise ValueError(f"Unknown method: {method}")
        

        chi_t = chi_t/(np.pi)
        
        return np.exp(-chi_t), omega_values, filter_values, integrand/(np.pi)
    

### 
# The rest of the functions in this document implement CPU parallelism for computing the coherence profile, C(t), via coherence_decay_profile_finite_peaks_with_widths
# Since, in general, we want to compute C(t) over an array of timepoints, this can be computationally annoying as the filter function, F(omega,t) changes for each time point,
# so, we must re-compute the integral for each of these timepoints. Fortunately, these computations are independent of one another, so we can parallelize across these timepoints, and 
# use every CPU available to us, rather than just one.
###
    
def retry_parallel_execution(func=None, max_retries=None, initial_n_jobs=None, min_n_jobs=1):
    """
    Decorator that implements retry logic with CPU reduction for parallel execution.
    Can be used with or without parameters.
    
    Args:
        func: The function to wrap
        max_retries: Maximum number of retry attempts (default: use all available CPUs down to min_n_jobs)
        initial_n_jobs: Initial number of CPUs to use (default: all available - 1)
        min_n_jobs: Minimum number of CPUs to try before giving up (default: 1)
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            # Allow overriding initial_n_jobs through function kwargs
            n_jobs = kwargs.pop('initial_n_jobs', None) or initial_n_jobs or max(1, os.cpu_count())
            if max_retries is None:
                max_retry_attempts = n_jobs - min_n_jobs + 1
            else:
                max_retry_attempts = max_retries
            
            attempt = 0
            current_n_jobs = n_jobs
            
            while attempt < max_retry_attempts and current_n_jobs >= min_n_jobs:
                try:
                    kwargs['n_jobs'] = current_n_jobs
                    # print(f"\nAttempting execution with {current_n_jobs} CPU{'s' if current_n_jobs > 1 else ''}...")
                    result = f(*args, **kwargs)
                    # print(f"Successfully completed using {current_n_jobs} CPU{'s' if current_n_jobs > 1 else ''}")
                    return result
                
                except Exception as e:
                    if "SIGKILL" in str(e) or isinstance(e, MemoryError):
                        attempt += 1
                        current_n_jobs -= 1
                        
                        if current_n_jobs >= min_n_jobs:
                            print(f"\nExecution failed with {current_n_jobs + 1} CPUs: {str(e)}")
                            print(f"Retrying with {current_n_jobs} CPU{'s' if current_n_jobs > 1 else ''}...")
                            time.sleep(2)  # Give system time to clean up resources
                        else:
                            print("\nReached minimum CPU count. Raising error.")
                            raise ParallelExecutionError(
                                f"Failed to execute even with minimum {min_n_jobs} CPU(s). Last error: {str(e)}"
                            )
                    else:
                        # If it's not a SIGKILL or MemoryError, re-raise the original exception
                        raise
            
            raise ParallelExecutionError(
                f"Exceeded maximum retry attempts ({max_retry_attempts}) or minimum CPU count reached"
            )
        return wrapper

    if func is None:
        return decorator
    return decorator(func)

@retry_parallel_execution
def parallel_coherence_decay(times, N, tau_p, method, noise_profile, *args, n_jobs=None, batch_size=None, max_memory_per_worker=1000, **kwargs):
    """
    Parallelized version of coherence_decay_profile_finite_peaks_with_widths. Computes the coherence decay profile, C(t), over an array of timepoints.
    The function defaults to using all available CPUs (n_jobs=None). You can change this parameter to use a specific number of CPUs.
    Inputs:
    times: a numpy array of time values
    n: a float of the number of pulses
    tau_p: a float of the pulse width
    method: a string of the method to use for integration. Options are "quad", "trapezoid", "simpson", and "log_sum_exp".
    noise_profile: a function that implements the noise spectrum, S(omega)
    *args: a list of arguments to pass to the noise_profile function
    n_jobs: an int of the number of CPUs to use for computation. Default is None, which uses all available CPUs.
    **kwargs: a dictionary of keyword arguments to pass to the function
    Output:
    C_t: a numpy array of the coherence decay profile, C(t)
    omega_values_list: a list of numpy arrays of angular frequencies used in the integration
    filter_values_list: a list of numpy arrays of the filter function values, evaluated at omega_values
    integrand_list: a list of numpy arrays of the integrand values given by S(omega)*F(omega*t)/(2*pi*omega**2), evaluated at omega_values
    """
    if not isinstance(N, int):
        raise TypeError(f"N must be an integer, got {type(N).__name__}")

    if n_jobs is None:
        n_jobs = -1  # -1 means using all available CPUs

    if batch_size is None:
        batch_size = max(1, len(times) // (n_jobs * 4))  # Divide work into smaller chunks

    # print(n_jobs)
    # if n_jobs >1:
    #     print(f"Using {n_jobs} CPUs with batch size {batch_size}")
    # elif n_jobs < 0:
    #     print(f"Using {os.cpu_count() + n_jobs + 1} CPU with batch size {batch_size}")

    time_batches = [times[i:i + batch_size] for i in range(0, len(times), batch_size)]

    def process_batch(batch):
        return [coherence_decay_profile_finite_peaks_with_widths(
            t, N, tau_p, method, noise_profile, *args, 
            max_memory_mb=max_memory_per_worker, **kwargs) 
                for t in batch]
    
    with parallel_backend('loky', n_jobs=n_jobs):
    # with parallel_backend('loky', n_jobs=n_jobs, inner_max_num_threads=1): # Use this line if you are using numpy or scipy functions that are not thread-safe
        results_nested = Parallel(verbose=0)(
            delayed(process_batch)(batch) 
            # for batch in tqdm(time_batches, desc="Processing batches")
            for batch in time_batches
        )
    
    # Flatten results and filter out None values
    results = [r for batch in results_nested for r in batch if r is not None]
    
    if not results:
        raise RuntimeError("No valid results were produced")
    
    # Unpack the results
    C_t, omega_values_list, filter_values_list, integrand_list = zip(*results)
    
    return (np.array(C_t), omega_values_list, filter_values_list, integrand_list)