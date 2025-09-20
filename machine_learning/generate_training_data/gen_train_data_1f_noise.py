import os
import sys
import psutil
from math import ceil, fsum
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time

import random

from scipy.signal import find_peaks
from scipy.integrate import quad, simpson, trapezoid, IntegrationWarning

from scipy.stats import cauchy

# import joblib
from joblib import Parallel, delayed, parallel_backend

# from tqdm import tqdm, trange
# from tqdm.contrib import tenumerate

from itertools import product

# import h5py
import pandas as pd
import fcntl
import time
from contextlib import contextmanager

import multiprocessing

# Custom Exception Classes
class ParallelExecutionError(Exception):
    """Custom exception for parallel execution failures"""
    pass

class MemoryThresholdError(Exception):
    """Custom exception for memory threshold exceeded"""
    pass

global_filepath = './training_data/'
# Check if the folder exists
if not os.path.exists(global_filepath):
    # Create the folder
    os.makedirs(global_filepath)
    print(f"Folder '{global_filepath}' created.")
else:
    print(f"Folder '{global_filepath}' already exists.")

#define the functions to have different kinds of noise
##### Document Conventions #####
# omega = 2 pi f
#t = 1/f

# I utilize numba just in time compilation throughout the code to speed up computation. you can see the @nb.njit decorator on top of the functions

@nb.njit(parallel=False)
def noise_spectrum_1f(omega, A, alpha):
    '''
    Implements S(omega) = A/f^alpha
    Inputs: 
    omega: a numpy array of angular frequencies
    A: a numpy array amplitude of the noise spectrum
    alpha: a numpy array of the exponent of the noise spectrum
    Output:
    S(omega) = A / (omega ** alpha)
    '''
    return np.divide(A, np.power(omega, alpha))

# I'm using the scipy.stats.cauchy.pdf function to compute the Lorentzian distribution., instead of using numba jit, because I didn't see much impact 
# on the performance of the function. The scipy function should be more stable and has better numerical precision, but I have also provided a numba implementation
# that should do the same thing (but please check).
def noise_spectrum_lor(omega, omega_0, gamma,A):
    '''
    Implements S(omega) = A * (1/(pi*gamma(1+((omega-omega_0)/gamma)**2)))
    Inputs:
    omega: a numpy array of angular frequencies
    omega_0: a float of the central frequency
    gamma: a float of the half width at half maximum
    A: a float of the amplitude of the noise spectrum
    Output:
    S(omega): a numpy array
    '''
    return np.multiply(A,cauchy.pdf(omega, loc=omega_0, scale=gamma))

# @nb.njit(parallel=False)
# def noise_spectrum_lor(omega, omega_0, gamma,A):
#     '''
#     Implements S(omega) = A * (1/(pi*gamma(1+((omega-omega_0)/gamma)**2)))
#     Inputs:
#     omega: a numpy array of angular frequencies
#     omega_0: a float of the central frequency
#     gamma: a float of the half width at half maximum
#     A: a float of the amplitude of the noise spectrum
#     Output:
#     S(omega): a numpy array
#     '''
#     # Preallocate the result array
#     result = np.empty_like(omega, dtype=np.float64)
    
#     # Compute Lorentzian (Cauchy) distribution manually
#     for i in range(len(omega)):
#         # Equivalent to scipy.stats.cauchy.pdf(x, loc=omega_0, scale=gamma)
#         result[i] = A / (np.pi * gamma * (1 + ((omega[i] - omega_0) / gamma)**2))
    
#     return result

@nb.njit(parallel=False)
def noise_spectrum_white(omega, C):
    '''Implements S(omega)
    Handles omega as numpy array, list, or single number
    Always returns a numpy array
    '''
    if isinstance(omega, (int, float)):
        return np.array([C])
    else:
        return np.full(len(omega), C)

def noise_spectrum_combination(omega,f_params,lor_params,white_params):
    '''Implements the combination of noise spectra
    Inputs:
    omega: a numpy array of angular frequencies
    f_params: a dictionary of the parameters for the 1/f noise spectrum
    lor_params: a dictionary of the parameters for the Lorentzian noise spectrum
    white_params: a dictionary of the parameters for the white noise spectrum
    Output:
    S(omega): a numpy array
    '''

    noise_specturm_list = []

    for my_dict in [f_params, lor_params, white_params]:

        if bool(my_dict) == True:
            for key,value in my_dict.items():
                if type(value) is int:
                    my_dict[key] = [value]

            first_length = len(next(iter(my_dict.values())))
            all_same_length = all(len(lst) == first_length for lst in my_dict.values())

            if not all_same_length:
                    raise ValueError('All the parameters in the dictionary must have the same length.')
        else:
            first_length = 0
        
        if my_dict == f_params:
            if first_length != 0:
                f_values  = list(zip(my_dict["A"], my_dict["alpha"]))
                for A, alpha in f_values:
                    noise_specturm_list.append(noise_spectrum_1f(omega, A, alpha))
            else:
                print(f'{f_params=}'.split('=')[0] + " dictionary is empty. Assuming there is no 1/f noise persent.")
        if my_dict == lor_params:
            if first_length != 0:
                lor_values = list(zip(my_dict["omega_0"], my_dict["gamma"], my_dict["A"]))
                for omega_0, gamma, A in lor_values:
                    noise_specturm_list.append(noise_spectrum_lor(omega, omega_0, gamma, A))
            else:
                print(f'{lor_params=}'.split('=')[0] + " dictionary is empty. Assuming there is no Lorentzian noise persent.")
        if my_dict == white_params:
            if first_length != 0:
                white_values = my_dict["C"]
                for C in white_values:
                    noise_specturm_list.append(noise_spectrum_white(omega, C))
            else:
                print(f'{white_params=}'.split('=')[0] + " dictionary is empty. Assuming there is no white noise persent.")

    # If there are numerical issues, try increasing the precision of the dtype (e.g. np.float64). If that doesn't work, try using math.fsum.
    #  You'll have to write a custom function like precise_complex_sum above, which handles complex arrays.
    noise_specturm = np.sum(np.array(noise_specturm_list),axis=0)

    return noise_specturm

# define the function to compute coherence decay using a delta function approximation
@nb.njit(parallel=False)
def coherence_decay_profile_delta(t,noise_profile):
    """
    Calculate the coherence decay profile based on the provided formula, e**(-chi(t)),
    uder the assumption that the filter function is a delta function.
    Inputs:
    noise_profile: a function that implements the noise spectrum, S(omega)
    t: a float of the time variable
    Output:
    e**(-chi(t))
    """
    # noise_profile = noise_spectrum_1f(np.pi/t, A, alpha)
    chi_t = t*noise_profile/np.pi
    return np.exp(-chi_t)

@nb.njit(parallel=False)
def filter_function_approx(omega, N, T):
    '''Implemnents an approximation of the filter function, F(omega*t) (eq #2 in [Meneses,Wise,Pagliero,et.al. 2022]).
    Approximation assumes pi-pulse are instantaneous (i.e. tau_pi = 0). Note, this function has singular points at (2m+1)n*Pi/t for integer m.
    Inputs:
    omega: a numpy array of angular frequencies
    N: a float of the number of pulses
    T: a float of the total time of the experiment
    Output:
    F(omega*t): a numpy array
    '''
    if not isinstance(N, int):
        raise TypeError(f"N must be an integer, got {type(N).__name__}")
    
    if N % 2 == 0:
        return 16 * (np.sin((omega*T) / 2) ** 2) * (np.sin((omega*T)/(4*N))**4) / (np.cos((omega*T)/(2*N))**2)
    else:
        return 16 * (np.cos((omega*T) / 2) ** 2) * (np.sin((omega*T)/(4*N))**4) / (np.cos((omega*T)/(2*N))**2)

@nb.njit(parallel=False)
def numba_complex_sum(t_k, omega):
    result = np.zeros_like(omega, dtype=np.complex128)
    for i in range(len(omega)):
        for k in range(len(t_k)):
            result[i] += ((-1)**(k+1))*np.exp(1j * omega[i] * t_k[k])
    return result

def filter_function_finite(omega, N, T, tau_p, method='numba'):
    ''' Implements the filter function, F(omega*t) (eq #1 in the paper).
    Inputs:
    omega: a numpy array of angular frequencies
    N: a float of the number of pulses
    T: a float of the total time for the experiment
    tau_p: a float of the pulse width
    Output:
    F(omega*t): a numpy array
    '''
    # Convert single number to array if needed
    single_number = isinstance(omega, (int, float))
    if single_number:
        omega = np.array([omega])

    if T < N * tau_p:
        raise ValueError("The total time of the experiment is less than the total time of the pulses.")
    else:
        # t_k = np.linspace((T/N-tau_p)/2, T*(1-1/(2*N))-(tau_p/2), N)  # Pulses are evenly spaced. This is the time of beginning of each pulse.
        t_k = T/(2*N)*np.arange(1,2*N+1,2) # Pulses are evenly spaced. This is the middle of the each pulse. Alternatively, np.linspace((T/N)/2, T*(1-1/(2*N)), N)
        if (t_k<0).any():
            raise ValueError("One of the Pulse start times is negative. This is not allowed.")
        
    if method == 'numba':
        # Uses numba to speed up the sum. Should be the fastest implementation.
        # Doesn't store the intermediate results, so it's memory efficient.
        sum_term = numba_complex_sum(t_k, omega)
    elif method == 'numpy_array':
        # Uses numpy array broadcasting to vectorize the sum. 
        # A little faster that a summation loop, and more numerically stable, but has a higher memory overhead. 
        sum_array = np.exp(1j * omega[:, np.newaxis] * t_k)
        sum_array[:,::2] *= -1 # Adds negative sign to odd indices of t_k
        # sum_array.sort(axis=1)
        sum_term = np.sum(sum_array, axis=1)
    elif method == "numpy":
        # Uses a for loop to sum the terms. Slowest implementation, but most more memory efficient than the array version.
        # You can instead sum the positive and negative terms separately, then subtract them, for a slightly more numerically stable result.

        # neg_sum_term = np.zeros(omega.shape)
        # pos_sum_term = np.zeros(omega.shape)
        # for k in range(0,N,2):
        #     # neg_sum_term = neg_sum_term - np.exp(1j * omega * t_k[k])
        #     neg_sum_term = np.sum(np.vstack((neg_sum_term,np.exp(1j * omega * t_k[k]))),axis=0)
            
        # for k in range(1,N+1,2):
        #     # pos_sum_term = pos_sum_term + np.exp(1j * omega * t_k[k])
        #     pos_sum_term = np.sum(np.vstack((pos_sum_term,np.exp(1j * omega * t_k[k]))),axis=0)

        # sum_term = (pos_sum_term-neg_sum_term)
            
        sum_term = np.zeros(omega.shape)
        for k in range(N):
            sum_term += ((-1)**(k+1)) * np.exp(1j * omega * t_k[k])
    
    result = np.power(np.abs(1 + np.power(-1,N+1) * np.exp(1j * omega * T) + 2 * np.cos(omega * tau_p / 2) * sum_term),2)
    
    return result

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

def coherence_decay_profile_finite_peaks_with_widths(t, N, tau_p, method, noise_profile, *args, max_memory_mb=15000, **kwargs):
    """Computes the integral in eq #1 in [Meneses,Wise,Pagliero,et.al. 2022] exactly, considering finite pulse widths, which computes the Coherence decay profile, C(t). 
    Inputs:
    t: a float of the time variable
    N: a float of the number of pulses
    tau_p: a float of the pulse width
    method: a string of the method to use for integration. Options are "quad,"trapezoid","simpson",and "log_sum_exp". 
    noise_profile: a function that implements the noise spectrum, S(omega)
    *args: a list of arguments to pass to the noise_profile function
    **kwargs: a dictionary of keyword arguments to pass to the function
    Output:
    np.exp(-chi_t): The coherence decay profile, C(t), the omega values used in the integration, the filter function values, and the integrand values.
    omega_values: a numpy array of angular frequencies
    filter_values: a numpy array of the filter function values, evaluated at omega_values
    integrand: a numpy array of the integrand values given by S(omega)*F(omega*t)/(2*pi*omega**2), evaluated at omega_values
    """
    if not (isinstance(N, int) and N > 0):
        raise TypeError(f"N must be a positive integer, got {type(N).__name__}")
    
    if kwargs.get('num_peaks_cutoff'):
        if not (isinstance(kwargs['num_peaks_cutoff'], int) and kwargs['num_peaks_cutoff'] >= 0):
            raise TypeError(f"num_peaks_cutoff must be a positive integer, got {type(kwargs['num_peaks_cutoff']).__name__}")
        
    current_memory = monitor_memory()
    if current_memory > max_memory_mb:
        raise MemoryError(f"Memory usage ({current_memory}MB) exceeded threshold")   

    try:
        omegas = np.logspace(np.log10(kwargs.get("omega_range")[0]), np.log10(kwargs.get("omega_range")[1]), 10**7) # setting resolution to 10^8 points. Might be a little overkill
    except KeyError:
        omegas = np.logspace(-4, 8, 10**7) # setting resolution to 10^8 points. Might be a little overkill
    S_w = noise_profile(omegas, *args)  # Check that the noise_profile function works with the omega_values

    k = 1000
    # Find the k largest values
    # k_largest_values = np.partition(S_w, -k)[-k:]
    k_largest_indices = np.argpartition(S_w, -k)[-k:]

    local_maxima_indices, _ =  find_peaks(S_w)
    local_minima_indices, _ = find_peaks(-S_w)

    points_of_interest = np.unique(np.concatenate((omegas[local_maxima_indices], omegas[local_minima_indices],omegas[k_largest_indices])))
    # print(len(points_of_interest))

    # Find the peaks of the filter function, located at omega = (2m+1)n*Pi/t for integer m
    base = np.pi * N / t

    omega_peaks = np.array([base * (2*k + 1) for k in range(100)]) # consider the first 100 FF resonance peaks

    # Check for additional FF resonance peaks near the points of interest on the noise
    # additional_filter_peaks = []
    # for point in points_of_interest:
    #     k = round((point/base - 1)/2)
    #     additional_filter_peaks.append(base * (2*k + 1))
    # additional_filter_peaks = np.array(additional_filter_peaks)

    k_values = np.round((points_of_interest/base - 1)/2).astype(int)
    additional_filter_peaks = base * (2*k_values + 1)

    try:
        omega_sweep = np.logspace(np.log10(kwargs.get("omega_range")[0]), np.log10(kwargs.get("omega_range")[1]), kwargs.get("omega_resolution",10**5)) # setting resolution to 10^8 points. Might be a little overkill
    except KeyError:
        omega_sweep = np.logspace(-4, 8, kwargs.get("omega_resolution",10**5)) # setting resolution to 10^8 points. Might be a little overkill

    breakpoints = np.unique(np.concatenate((omega_peaks, additional_filter_peaks, points_of_interest)))
    omega_values = np.unique(np.concatenate((breakpoints, omega_sweep)))


    filter_values = filter_function_finite(omega_values, N, t, tau_p)

    integrand = np.multiply(filter_values, np.divide(noise_profile(omega_values, *args),(np.power(omega_values,2))))

    largest_peak_indices, _ = find_k_largest_peaks(integrand, kwargs.get('num_peaks_cutoff', len(omega_values)))

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
                # additional_omega_values.append(omega_values, np.logspace(np.log10(peak-peak*peak_width), np.log10(peak+peak*peak_width), peak_res))
                additional_omega_values = np.append(additional_omega_values,np.linspace(peak-peak*peak_width, peak+peak*peak_width, peak_res))
                
        additional_omega_values = additional_omega_values[additional_omega_values > 0] # Remove non-positive values to be safe. Important when N=1 to avoid devide by zero errors.
        additional_filter_values = filter_function_finite(additional_omega_values, N, t, tau_p)
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

    if method == "quad":
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
                    print(f"\nAttempting execution with {current_n_jobs} CPU{'s' if current_n_jobs > 1 else ''}...")
                    result = f(*args, **kwargs)
                    print(f"Successfully completed using {current_n_jobs} CPU{'s' if current_n_jobs > 1 else ''}")
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
    Parallelized version of coherence_decay_profile_finite_peaks_with_widths.
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

    # if n_jobs is None:
    #     n_jobs = -1  # -1 means using all available CPUs

    if batch_size is None:
        batch_size = max(1, len(times) // (n_jobs * 4))  # Divide work into smaller chunks

    print(n_jobs)
    if n_jobs >1:
        print(f"Using {n_jobs} CPUs with batch size {batch_size}")
    elif n_jobs < 0:
        print(f"Using {os.cpu_count() + n_jobs + 1} CPU with batch size {batch_size}")

    time_batches = [times[i:i + batch_size] for i in range(0, len(times), batch_size)]

    def process_batch(batch):
        return [coherence_decay_profile_finite_peaks_with_widths(
            t, N, tau_p, method, noise_profile, *args, 
            max_memory_mb=max_memory_per_worker, **kwargs) 
                for t in batch]
    
    with parallel_backend('loky', n_jobs=n_jobs):
    # with parallel_backend('loky', n_jobs=n_jobs, inner_max_num_threads=1): # Use this line if you are using numpy or scipy functions that are not thread-safe
        results_nested = Parallel(verbose=1)(
            delayed(process_batch)(batch) 
            for batch in time_batches
        )
    
    # Flatten results and filter out None values
    results = [r for batch in results_nested for r in batch if r is not None]
    
    if not results:
        raise RuntimeError("No valid results were produced")
    
    # Unpack the results
    C_t, omega_values_list, filter_values_list, integrand_list = zip(*results)
    
    return (np.array(C_t), omega_values_list, filter_values_list, integrand_list)

##### GENERATE TRAINING SET #####

def generate_training_data_delta(t_points, noise_profile, *args):

    parameters = list(product(*args))
    C_t_list = []

    for params in parameters:
        noise = noise_profile(np.pi/t_points, *params)
        C_t = coherence_decay_profile_delta(t_points, noise)
        C_t_list.append([C_t,noise,params])
    
    return C_t_list

# @retry_parallel_execution
def generate_training_data_finite(times,parameters, n, tau_p, num_cpus, method, noise_profile, **integration_params):
    """
    Generate coherence decay data with parallelized computation over both time points and parameters.
    
    Parameters:
    -----------
    t_points : array-like
        Time points at which to compute coherence
    n : int
        Number of pulses
    tau_p : float
        Pulse width
    method : str
        Integration method ('quad', 'simpson', 'trapezoid')
    noise_profile : callable
        Function to compute noise spectrum
    *args : iterables
        Parameters to sweep over for parameter space exploration
    **integration_params : dict
        Additional parameters for integration (e.g., omega_range, omega_resolution)
    
    Returns:
    --------
    list of [C_t, noise_spectrum, params]
        C_t: numpy array of coherence values for each time point
        noise_spectrum: numpy array of noise spectrum
        params: specific parameter combination
    """
    
    # Prepare omega values for noise spectrum
    try:
        omega = np.logspace(
            np.log10(integration_params.get("omega_range")[0]), 
            np.log10(integration_params.get("omega_range")[1]), 
            integration_params.get("omega_resolution", 10**5)
        )
    except KeyError:
        omega = np.logspace(-4, 8, integration_params.get("omega_resolution", 10**5))
    
    C_t_list = []

    for params in parameters:
        if int(num_cpus) != 1:
            C_t, _, _, _ = parallel_coherence_decay(times, n, tau_p, method, noise_profile, *params, initial_n_jobs=int(num_cpus), max_memory_per_worker=10**4, **integration_params)
        else:
            C_t, _, _, _ = zip(*[coherence_decay_profile_finite_peaks_with_widths(t, n, tau_p, method, noise_profile, *params,**integration_params) for t in times])
        C_t_list.append([C_t,noise_profile(omega, *params),params])

    # for params in parameters:
    #     if int(os.getenv("SLURM_CPUS_PER_TASK")) > 1:
    #         C_t, _, _, _ = parallel_coherence_decay(times, n, tau_p, method, noise_profile, *params, initial_n_jobs=int(os.getenv("SLURM_CPUS_PER_TASK", 1)), max_memory_per_worker=10**4, **integration_params)
    #     else:
    #         C_t, _, _, _ = zip(*[coherence_decay_profile_finite_peaks_with_widths(t, n, tau_p, method, noise_profile, *params,**integration_params) for t in times])
    #     C_t_list.append([C_t,noise_profile(omega, *params),params])

    return C_t_list

@contextmanager
def file_lock(filepath):
    """Context manager for file locking to handle concurrent access."""
    lockfile = f"{filepath}.lock"
    with open(lockfile, 'w') as f:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError:
                time.sleep(1)  # Wait before retrying
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def make_and_save_training_data(t_points, save_every_n, num_cpus, noise_profile, *args, **kwargs):
    """
    Generate and save training data, splitting parameters into groups for intermittent saving.

    Parameters:
    -----------
    t_points : array-like
        Time points for the training data.
    assume_delta : bool
        Whether to assume delta function noise.
    noise_profile : callable
        Function to compute noise spectrum.
    *args : iterables
        Parameter ranges for exploration.
    **kwargs : dict
        Additional arguments for data generation.

    Returns:
    --------
    None
    """
    # Create a filename appendage based on the noise profile and parameter characteristics
    if noise_profile.__name__ == "noise_spectrum_combination":
        appendage = (
            "num_1f_" + str(len(args[0][0]["A"])) +
            "_num_lor_" + str(len(args[1][0]["gamma"])) +
            "_num_white_" + str(len(args[2][0]["C"]))
        )
    else:
        appendage = ""

    # Define file names
    params = '_'.join(f"{k}_{v}" for k, v in kwargs.items())
    filename = f"{noise_profile.__name__}_{params}"
    hdf_filename = f'training_data_{filename}{appendage}_finite_width.h5'
    output_path = os.path.join(global_filepath, hdf_filename)

    # Initialize existing parameters set
    existing_params = set()

    n = kwargs.pop('N', None) 
    tau_p = kwargs.pop('tau_p', None)
    integration_method = kwargs.pop('integration_method', None)

    print(f"Checking existing data in {hdf_filename}")

    # Check existing data with file locking
    with file_lock(output_path):
        try:
            with pd.HDFStore(output_path, mode='r') as store:
                if 'noise_profile_args' in store:
                    existing_params.update(tuple(p) for p in store['noise_profile_args'].values)
            print(f"Found {len(existing_params)} existing parameter sets.")
        except (FileNotFoundError, KeyError):
            print(f"No existing data file found.")

    # Generate all possible parameters and remove existing ones
    all_parameters = list(product(*args))
    parameters = [param for param in all_parameters if tuple(param) not in existing_params]

    if not parameters:
        print(f"All parameters already exist in the dataset. Nothing to compute.")
        return
        
    print(f"After removing existing parameters, {len(parameters)} parameters remain to be computed.")

# Define batch size for parameter groups
    batch_size = save_every_n
    num_batches = (len(parameters) + batch_size - 1) // batch_size

    print(f"Generating data for {len(parameters)} new parameters.")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(parameters))
        param_subset = parameters[start_idx:end_idx]

        # Generate data for the current subset
        data_subset = generate_training_data_finite(t_points, param_subset, n, tau_p, num_cpus, 
                                                  integration_method, noise_profile, **kwargs)

        # Prepare batch data
        batch_X = []
        # batch_Y = []
        batch_params = []
        for item in data_subset:
            batch_X.append(item[0]) # C_t values
            # batch_Y.append(item[1])  # Noise profiles
            batch_params.append(item[2]) # Parameters

        # Save batch with file locking
        with file_lock(output_path):
            with pd.HDFStore(output_path, mode='a', complevel=9, complib='zlib') as store:
                # Read existing data
                try:
                    existing_X = store['data']
                    existing_params = store['noise_profile_args']
                except KeyError:
                    existing_X = pd.DataFrame()
                    existing_params = pd.DataFrame()

                # Append new data
                new_X = pd.DataFrame(batch_X)
                # new_Y = pd.DataFrame(batch_Y)
                new_params = pd.DataFrame(batch_params)
                
                store['data'] = pd.concat([existing_X, new_X], ignore_index=True)
                store['noise_profile_args'] = pd.concat([existing_params, new_params], ignore_index=True)
                
                # Store metadata if not exists
                if 'noise_profile' not in store:
                    store['noise_profile'] = pd.Series([str(noise_profile.__name__)])
                if 'filter_function_args' not in store:
                    store['filter_function_args'] = pd.Series([kwargs])

        print(f"Finished and saved batch {batch_idx + 1}/{num_batches}.")

    print(f"Completed all batches.")


np_data_type = np.float64
# Parameters

num_samples = 2*10**4
# 1/f noise parameters
A_values = np.linspace(0, 100, ceil(np.sqrt(num_samples/2)),dtype=np_data_type)  # Adjust the number of points as needed
A_values = np.append(A_values, np.linspace(0, 5, ceil(np.sqrt(num_samples/2)),dtype=np_data_type))
alpha_values = np.linspace(0, 3, ceil(np.sqrt(num_samples)),dtype=np_data_type)  # Adjust the number of points as needed
# Ensures that the parameters are all unique , and sorted. Not necessary, but convenient.
A_values = np.unique(A_values)[::-1]
alpha_values = np.unique(alpha_values)[::-1]
f_args = [A_values, alpha_values]


finite_width_params = {
    "N": int(sys.argv[1]), #CPMG-N (test spin echo)
    "tau_p": float(sys.argv[2]), #pi pulse width in mircoseconds
    "integration_method": "trapezoid", # The method to use for integration. Options are "quad","trapezoid",and "simpson".
    "omega_resolution": int(10**5), # 6*10**5 The number of points to use when integrating to yield the coherence decay profile, C(t), unless method is "quad", in which case, this is the resolution used to determine the filter function and integrand outputs.
    "omega_range" : (10**(-4), 10**8), # The number of peaks to use in the filter function
    "num_peaks_cutoff": 100, # must be >=0 and <num_peaks. If this value is set, then only the top num_peaks_cutoff peaks will be used in the integration. This can be useful for speeding up the integration when the noise spectrum has many peaks.
    "peak_resolution": 100, # For every method except "quad", you can specify additional points around the singular points of the filter function. This argument specifies the number of points to use around each singular point.
    # "quad_limit": 10**5, # The maximum number of subdivisions in the adaptive quadrature algorithm
    # "epsabs": 1.49e-8, # Absolute tolerance for the quadrature integration
    # "epsrel": 1.49e-8, # Relative tolerance for the quadrature integration
}

print("##################")
print(f"Number of pulses: {finite_width_params['N']}")
print(f"Pulse width: {finite_width_params['tau_p']} microseconds")
print("##################")

t_points = np.logspace(np.log10(finite_width_params["N"]*finite_width_params["tau_p"]+1e-10), np.log10(300), 201, dtype=np_data_type)  # Time points for coherence profile. Units are Microseconds
# frequency_points = np.pi / t_points  # Frequency range for generating noise spectrum

# print("Total Tasks: ",int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1)))
# print("SLURM Task ID: ",int(os.getenv("SLURM_ARRAY_TASK_ID", 0)))

save_every_n = np.max((int(num_samples/100),1)) # The number of parameter sets to process in each batch. For intermittent saving of results.
# save_every_n = 3
num_cpus = multiprocessing.cpu_count()
print("Generating Training Data: 1/f Noise")
print(finite_width_params)
print()
print(f_args)
print("****************************")
make_and_save_training_data(t_points,save_every_n,num_cpus,noise_spectrum_1f,*f_args,**finite_width_params)
