import numpy as np
import numba as nb
from scipy.stats import cauchy

#define the functions to have different kinds of noise
##### Document Conventions #####
# omega = 2 pi f
# t = 1/f

# This file defines the various noise spectra functions that can be used to model different types of noise in a system.
# The functions are designed to handle numpy arrays of angular frequencies, but can also handle single values.
# The functions return numpy arrays of the noise spectrum values.
# The function noise_spectrum_combination allows for the combination of different noise spectra so that the result is a sum of the individual spectra. 
# You can even specify multiple instances of the same noise type, e.g. two 1/f noise spectra with different amplitudes and exponents.

# If you add a new noise spectrum, I reccommend you to add it to the noise_spectrum_combination function, and pass that to any of the C(t) functions, so that you can use it in the same way as the other noise spectra.

# I utilize numba just in time compilation throughout the code to speed up computation. you can see the @nb.njit decorator on top of the functions

@nb.njit(parallel=False)
def noise_spectrum_1f(omega, A, alpha):
    '''
    Implements S(ω) = A/f^α
    Inputs: 
    omega: a numpy array of angular frequencies
    A: a numpy array amplitude of the noise spectrum
    alpha: a numpy array of the exponent of the noise spectrum
    Output:
    S(ω)= A / (ω ** α)
    '''
    return np.divide(A, np.power(omega, alpha))

# I'm using the scipy.stats.cauchy.pdf function to compute the Lorentzian distribution., instead of using numba jit, because I didn't see much impact 
# on the performance of the function. The scipy function should be more stable and has better numerical precision, but I have also provided a numba implementation
# that should do the same thing (but please check).
def noise_spectrum_lor(omega, omega_0, gamma,A):
    '''
    Implements S(ω) = A * (1/(pi*gamma(1+((omega-omega_0)/gamma)**2)))
    Inputs:
    omega: a numpy array of angular frequencies
    omega_0: a float of the central frequency
    gamma: a float of the half width at half maximum
    A: a float of the amplitude of the noise spectrum
    Output:
    S(ω): a numpy array
    '''
    return np.multiply(A, cauchy.pdf(omega, loc=omega_0, scale=gamma))

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
    '''Implements S(ω)
    Handles omega as numpy array, list, or single number
    Always returns a numpy array
    '''
    if isinstance(omega, (int, float)):
        return np.array([C])
    else:
        return np.full(len(omega), C)

@nb.njit(parallel=False)
def noise_spectrum_double_power_law(omega, A, alpha, beta, gamma):
    """
    Calculate the continued fraction: A / omega^alpha * (1 + (omega/gamma)^(beta-alpha))

    Parameters:
    omega: a numpy array of angular frequencies
    A: numerator of the main fraction
    alpha: exponent applied to the inner fraction
    beta: 2nd exponent applied to the inner fraction. Must be greater than alpha. 
    gamma: denominator of the inner fraction. The cutoff frequency that indicates the transition from alpha to beta behavior.

    Returns:
    The value of S(ω) (real array)
    """

    assert beta > alpha, f"beta ({beta}) must be greater than alpha ({alpha})"
    
    inner_fraction = (omega) / gamma

    power_result = inner_fraction**(beta-alpha)
    
    denominator = 1 + power_result
    result = A / (denominator*omega**alpha)

    return result

def noise_spectrum_combination(omega,f_params,lor_params,white_params, double_power_law_params):
    '''Implements the combination of noise spectra
    Inputs:
    omega: a numpy array of angular frequencies
    f_params: a dictionary of the parameters for the 1/f noise spectrum. e.g. For two 1/f profiles, {"alpha": [1,2], "A": [1,2]}
    lor_params: a dictionary of the parameters for the Lorentzian noise spectrum. e.g. for 3 lorentzian profiles {"omega_0": [1,2,3], "gamma": [1,2,3], "A": [1,2,3]}
    white_params: a dictionary of the parameters for the white noise spectrum. e.g. {"C": [1]}
    double_power_law_params: a dictionary of the parameters for the double power law noise spectrum. e.g. {"A": [1], "alpha": [1], "beta": [2], "gamma": [1]}
    Output:
    S(omega): a numpy array
    '''

    noise_specturm_list = []

    for i,my_dict in enumerate([f_params, lor_params, white_params, double_power_law_params]):

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
        
        # if (my_dict == f_params and i==0):
        if i == 0:
            if first_length != 0:
                f_values  = list(zip(my_dict["A"], my_dict["alpha"]))
                for A, alpha in f_values:
                    noise_specturm_list.append(noise_spectrum_1f(omega, A, alpha))
            else:
                # print(f'{f_params=}'.split('=')[0] + " dictionary is empty. Assuming there is no 1/f noise persent.")
                pass
        # elif (my_dict == lor_params and i==1):
        elif i== 1:
            if first_length != 0:
                lor_values = list(zip(my_dict["omega_0"], my_dict["gamma"], my_dict["A"]))
                for omega_0, gamma, A in lor_values:
                    noise_specturm_list.append(noise_spectrum_lor(omega, omega_0, gamma, A))
            else:
                # print(f'{lor_params=}'.split('=')[0] + " dictionary is empty. Assuming there is no Lorentzian noise persent.")
                pass
        # elif (my_dict == white_params and i==2):
        elif i ==2:
            if first_length != 0:
                white_values = my_dict["C"]
                for C in white_values:
                    noise_specturm_list.append(noise_spectrum_white(omega, C))
            else:
                # print(f'{white_params=}'.split('=')[0] + " dictionary is empty. Assuming there is no white noise persent.")
                pass
        elif i == 3:
            if first_length != 0:
                combined_frac_values = list(zip(my_dict["A"], my_dict["alpha"], my_dict["beta"], my_dict["gamma"]))
                for A, alpha, beta, gamma in combined_frac_values:
                    noise_specturm_list.append(noise_spectrum_double_power_law(omega, A, alpha, beta, gamma))
            else:
                # print(f'{combined_frac_params=}'.split('=')[0] + " dictionary is empty. Assuming there is no continued fraction noise persent.")
                pass


    # If there are numerical issues, try increasing the precision of the dtype (e.g. np.float64). If that doesn't work, try using math.fsum.
    #  You'll have to write a custom function like precise_complex_sum above, which handles complex arrays.
    noise_specturm = np.sum(np.array(noise_specturm_list),axis=0)

    return noise_specturm