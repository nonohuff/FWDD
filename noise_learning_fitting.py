import json
import os
import psutil
import time
import warnings
import sys
from math import ceil, fsum
import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

from scipy import interpolate
from scipy.signal import find_peaks
from scipy.integrate import quad, simpson, trapezoid, IntegrationWarning
from scipy.optimize import curve_fit, minimize, differential_evolution, brentq, NonlinearConstraint, least_squares
from scipy.stats import cauchy, chi2
from scipy.signal import butter, filtfilt
from scipy.special import huber

from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.utils import resample

from skopt import gp_minimize
from skopt.space import Real

from tqdm.notebook import tqdm, trange
from tqdm.contrib import tenumerate

import joblib
from joblib import Parallel, delayed, parallel_backend

from itertools import product, combinations
import multiprocessing

from package import filter_function as ff
from package import noise_spectra as ns
from package import coherence_profile as cp
from package.noise_learning_fitting import func_to_fit, fit_coherence_decay, fit_noise_spectrum, fit_coherence_decay_combined, create_parameter_constraints
from package.fitting_utils import find_widest_contiguous_stretch, find_time_range_for_C_t_bounds, add_gaussian_noise, format_parameters, create_combined_analysis_plot, calculate_total_combinations, bootstrap_multiple_samples, analyze_intervals



experimental_data = sys.argv[12].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh'] if len(sys.argv) > 12 else False # If True, loads experimental data from json file. If False, uses artificially simulated data.

### Dynamical Decoupling sequence parameters ###
n_values = [1,8,128,256,512]
tau_p = float(sys.argv[1]) # pi pulse width in microseconds. The filter function code assumes equally-spaced pi pulses with finite width tau_p.

### Finite-width filter function and integration parameters ###
integration_method = "trapezoid" # The method to use for integration. Options are "quad","trapezoid",and "simpson".
omega_resolution = int(6*10**5) # The number of points to use when integrating to yield the coherence decay profile, C(t), unless method is "quad", in which case, this is the resolution used to determine the filter function and integrand outputs.
omega_range = (10**(-4), 10**8) # The number of peaks to use in the filter function
num_peaks_cutoff = 100 # Number of peaks to use in the integration. Must be >=0. If this value is set, then only the top num_peaks_cutoff peaks will be used in the integration. This can be useful for speeding up the integration when the noise spectrum has many peaks.
peak_resolution = 100 # For every method except "quad", you can specify additional points around the singular points of the filter function. This argument specifies the number of points to use around each singular point.

### Noise model parameters ###
# FYI each of these can be any non-negative integer, including zero.
N1f = int(sys.argv[2]) # Number of 1/f noise profiles present. Each profile has parameters A and alpha. So, a total of 2*N1f parameters.
Nlor = int(sys.argv[3]) # Number of Lorentzian noise profiles present. Each profile has parameters omega_0, gamma, and A. So, a total of 3*Nlor parameters.
NC = int(sys.argv[4]) # Number of white noise profiles present. Each profile has parameter C. So, a total of NC parameters. Since white noise is flat, it is equivalent to a single white noise profile with C equal to the sum of all individual C values, so you should never need to set NC to be greater than 1.
Ndpl = int(sys.argv[5]) # Number of double power law noise profiles present. Each profile has parameters A, alpha, beta, gamma. So, a total of 4*Ndpl parameters.

# Set this to True to use combined loss across all n values in the Dynamical Decoupling sequence. If True, finds the noise profile that best fits all C_n(t) values simultaneously. If False, fits each n value separately.
combined_loss = sys.argv[13].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh'] if len(sys.argv) > 13 else True  # Set to False to use original separate fitting

### Optimization options ###
meth = "diff_ev"  # Options: "L-BFGS-B", "diff_ev", "gp_minimize"
loss_type = "mse"  # Options: "mse", "huber"
delta_approx = sys.argv[6].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh'] if len(sys.argv) > 6 else False # Use delta approximation for the noise spectrum, i.e. S(omega) = A*delta(omega - n*pi/tau_p)

# whether to include Gaussian noise in the simulated C(t) data, and if so, the strength of that noise.
noise = sys.argv[7].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh'] if len(sys.argv) > 7 else False
noise_lvl = float(sys.argv[8]) if len(sys.argv) > 8 else 0.04

# Scaling factor for C(t). F=1 Implies that C(t) begins at 1. You can change this to account for experimental imperfections that reduce the initial coherence. For example, if the initial coherence is 0.9, set F=0.9.
F = float(sys.argv[11]) if len(sys.argv) > 11 else 1

# Set the range of C(t). If 0 < eps < 1, then C_t_max = 1-eps and C_t_min = eps. Otherwise, C_t_max = 2 and C_t_min = -2, Which just considers the full range of 0 < C(t) < 1.

# eps = 0
# if 0 < eps < 1:
#     C_t_max = 1 - eps
#     C_t_min = eps
# else:
#     C_t_max = 2
#     C_t_min = -2

C_t_min = float(sys.argv[14]) if len(sys.argv) > 14 else -2.0
C_t_max = float(sys.argv[15]) if len(sys.argv) > 15 else 2.0


############################################################################
print("Using experimental data:", experimental_data)
print("Combined Loss:", combined_loss)
print("Optimization Method:", meth)
print("Loss Function Type:", loss_type)
print("Delta Approximation:", delta_approx)

print("Included Gaussian Noise:", noise)
if noise:
    print("Noise Level:", noise_lvl)

print(f"C(t) scaling factor, F = {F}")


if 0 < C_t_max < 1:
    print_C_t_max = C_t_max
else:
    print_C_t_max = 1
if 0 < C_t_min < 1:
    print_C_t_min = C_t_min
else:
    print_C_t_min = 0
print(f"C(t) Range: [{print_C_t_min}, {print_C_t_max}]")

results_dict = {}


print("N values to fit:", n_values)


if experimental_data:
    global_filepath = './fitting/experimental_data_fits/'
else:
    global_filepath = './fitting/optimization_fits/'

finite_width_params = {
    "N1f": N1f, # Number of 1/f noise parameters
    "Nlor": Nlor, # Number of Lorentzian noise parameters
    "NC": NC, # Number of white noise parameters
    "Ndpl": Ndpl, # Number of double power law noise parameters
    "tau_p": tau_p, #pi pulse width in mircoseconds
    "integration_method": integration_method, # The method to use for integration. Options are "quad","trapezoid",and "simpson".
    "omega_resolution": omega_resolution, # 6*10**5 The number of points to use when integrating to yield the coherence decay profile, C(t), unless method is "quad", in which case, this is the resolution used to determine the filter function and integrand outputs.
    "omega_range" : omega_range, # The number of peaks to use in the filter function
    "num_peaks_cutoff": num_peaks_cutoff, # must be >=0 and <num_peaks. If this value is set, then only the top num_peaks_cutoff peaks will be used in the integration. This can be useful for speeding up the integration when the noise spectrum has many peaks.
    "peak_resolution": peak_resolution, # For every method except "quad", you can specify additional points around the singular points of the filter function. This argument specifies the number of points to use around each singular point.
    # Additional parameters for the "quad" integration method. Not used here since we are using Riemann sum methods.
    # "quad_limit": 10**5, # The maximum number of subdivisions in the adaptive quadrature algorithm
    # "epsabs": 1.49e-8, # Absolute tolerance for the quadrature integration
    # "epsrel": 1.49e-8, # Relative tolerance for the quadrature integration
}

# Extract counts
N1f = finite_width_params["N1f"]
Nlor = finite_width_params["Nlor"] 
NC = finite_width_params["NC"]
Ndpl = finite_width_params["Ndpl"]

# Initialize noise_params list with 4 empty dictionaries
noise_params = [{}, {}, {}, {}]

# Generate 1/f noise parameters
if N1f > 0:
    A_1f = np.random.choice(np.linspace(0, 100, 1001), size=N1f)
    alpha_1f = np.random.choice(np.linspace(0, 3, 1001), size=N1f)
    noise_params[0] = {"A": A_1f.tolist(), "alpha": alpha_1f.tolist()}

# Generate Lorentzian noise parameters  
if Nlor > 0:
    omega_0_lor = np.random.choice(np.linspace(1e-3, 1e8, 10001), size=Nlor)  # Central frequency range
    gamma_lor = np.random.choice(np.linspace(1e-4, 1e6, 10001), size=Nlor)    # Half-width range
    A_lor = np.random.choice(np.linspace(0, 100, 1001), size=Nlor)            # Amplitude range
    noise_params[1] = {
        "omega_0": omega_0_lor.tolist(), 
        "gamma": gamma_lor.tolist(), 
        "A": A_lor.tolist()
    }

# Generate white noise parameters
if NC > 0:
    C_white = np.random.choice(np.linspace(0, 5, 1001), size=NC)
    noise_params[2] = {"C": C_white.tolist()}

# Generate combined fraction noise parameters
if Ndpl > 0:
    A_frac = np.random.choice(np.linspace(0, 1e2, 1001), size=Ndpl)
    alpha_frac = np.random.choice(np.linspace(0, 4, 1001), size=Ndpl)
    beta_frac = np.random.choice(np.linspace(0, 1e4, 10001), size=Ndpl)
    gamma_frac = np.random.choice(np.linspace(1e-4, 1e2, 1001), size=Ndpl)
    noise_params[3] = {
        "A": A_frac.tolist(),
        "alpha": alpha_frac.tolist(), 
        "beta": beta_frac.tolist(),
        "gamma": gamma_frac.tolist()
    }

# Print the generated parameters for verification
print("Generated noise_params:")
for i, param_dict in enumerate(noise_params):
    param_names = ["f_params", "lor_params", "white_params", "double_power_law_params"]
    print(f"{param_names[i]}: {param_dict}")
    
# # You can also print individual parameter values like in your original code
# if noise_params[0]:  # If 1/f params exist
#     print(f"\n1/f noise - A values: {noise_params[0]['A']}")
#     print(f"1/f noise - alpha values: {noise_params[0]['alpha']}")
    
# if noise_params[2]:  # If white noise params exist  
#     print(f"White noise - C values: {noise_params[2]['C']}")
    
# if noise_params[1]:  # If Lorentzian params exist
#     print(f"Lorentzian noise - omega_0 values: {noise_params[1]['omega_0']}")
#     print(f"Lorentzian noise - gamma values: {noise_params[1]['gamma']}")
#     print(f"Lorentzian noise - A values: {noise_params[1]['A']}")
    
# if noise_params[3]:  # If double power law params exist
#     print(f"Double power law - A values: {noise_params[3]['A']}")
#     print(f"Double power law - alpha values: {noise_params[3]['alpha']}")
#     print(f"Double power law - beta values: {noise_params[3]['beta']}")
#     print(f"Double power law - gamma values: {noise_params[3]['gamma']}")

# First pass: Generate C_t_observed for all n values
C_t_observed_dict = {}
times_dict = {}
fixed_kwargs_dict = {}


############################################################################
# Example of what to change to do Bootstrapping.
# You may need to change some things, as this is from old code, but the main idea is:
# 1. experimental_data = False, noise =True, and noise_level = sigma.
# 2. instead of generating random parameters for C(t), use the parameters you found from fitting your experimental data.
############################################################################

# # Bootstrap the noise parameters
# C_t_sigmas = {1: 0.06138902539286488, 
#  8: 0.03925182944743504, 
#  128: 0.10197957240006385, 
#  256: 0.07537532562991836, 
#  512: 0.16274530391838227}

# ############################################################################################
# experimental_data = False
# print("Using experimental data:", souvik_data)

# N1f = 1
# Nlor = 0
# NC = 0
# Ncom_frac = 0

# N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
# n_values = [N]
# print("N values to fit:", n_values)

# # NEW OPTION: Set this to True to use combined loss across all n values
# combined_loss = False  # Set to False to use original separate fitting
# print("Combined Loss:", combined_loss)

# # Optimization options
# meth = "diff_ev" # Default optimization method (other options: "gp_minimize", "L-BFGS-B")
# print("Optimization Method:", meth)
# loss_type = "mse" # Default loss function type (other options: "huber")
# print("Loss Function Type:", loss_type)
# delta_approx = False # Use delta approximation for the noise spectrum, i.e. S(omega) = A*delta(omega - n*pi/tau_p)
# print("Delta Approximation:", delta_approx)

# noise = True
# noise_lvl = C_t_sigmas[N]
# print("Included Gaussian Noise:", noise)
# if noise:
#     print("Noise Level:", noise_lvl)

# F = 1
# print(f"C(t) scaling factor, F = {F}")

# print("Maximum number of iterations:", sys.argv[2] if len(sys.argv) > 2 else "50")
# print("Population size:", sys.argv[3] if len(sys.argv) > 3 else "10")


# # C_t_max = 0.95
# # C_t_min = 0.05
# # C_t_max = 2
# # C_t_min = -2

# C_t_min = -2
# C_t_max = 2

# if 0 < C_t_max < 1:
#     print_C_t_max = C_t_max
# else:
#     print_C_t_max = 1
# if 0 < C_t_min < 1:
#     print_C_t_min = C_t_min
# else:
#     print_C_t_min = 0
# print(f"C(t) Range: [{print_C_t_min}, {print_C_t_max}]")

# results_dict = {}

# global_filepath = './bootstrapping/'

# finite_width_params = {
#     "N1f": N1f, # Number of 1/f noise parameters
#     "Nlor": Nlor, # Number of Lorentzian noise parameters
#     "NC": NC, # Number of white noise parameters
#     "Ncf": Ncom_frac, # Number of combined fraction noise parameters
#     "N": 1, #CPMG-N (test spin echo)
#     "tau_p": 0.024, #pi pulse width in mircoseconds
#     "integration_method": "trapezoid", # The method to use for integration. Options are "quad","trapezoid",and "simpson".
#     "omega_resolution": int(10**5), # 6*10**5 The number of points to use when integrating to yield the coherence decay profile, C(t), unless method is "quad", in which case, this is the resolution used to determine the filter function and integrand outputs.
#     "omega_range" : (10**(-4), 10**8), # The number of peaks to use in the filter function
#     "num_peaks_cutoff": 100, # must be >=0 and <num_peaks. If this value is set, then only the top num_peaks_cutoff peaks will be used in the integration. This can be useful for speeding up the integration when the noise spectrum has many peaks.
#     "peak_resolution": 100, # For every method except "quad", you can specify additional points around the singular points of the filter function. This argument specifies the number of points to use around each singular point.
#     # "quad_limit": 10**5, # The maximum number of subdivisions in the adaptive quadrature algorithm
#     # "epsabs": 1.49e-8, # Absolute tolerance for the quadrature integration
#     # "epsrel": 1.49e-8, # Relative tolerance for the quadrature integration
# }

# # Extract counts
# N1f = finite_width_params["N1f"]
# Nlor = finite_width_params["Nlor"] 
# NC = finite_width_params["NC"]
# Ncom_frac = finite_width_params["Ncf"]


# FWFF_params = {1: [{"A": [24.53], "alpha": [0.73]}, {}, {}, {}],
# 8: [{"A": [16.18], "alpha": [0.69]}, {}, {}, {}],
# 128: [{"A": [143334052.69], "alpha": [5.2]}, {}, {}, {}],
# 256: [{"A": [31.41], "alpha": [1.24]}, {}, {}, {}],
# 512: [{"A": [118392.921361], "alpha": [3.386887]}, {}, {}, {}]
# }

# noise_params = FWFF_params[N] if N in FWFF_params else [{}, {}, {}, {}]

# # Print the generated parameters for verification
# print("Generated noise_params:")
# for i, param_dict in enumerate(noise_params):
#     param_names = ["f_params", "lor_params", "white_params", "combined_frac_params"]
#     print(f"{param_names[i]}: {param_dict}")
    
# # # You can also print individual parameter values like in your original code
# # if noise_params[0]:  # If 1/f params exist
# #     print(f"\n1/f noise - A values: {noise_params[0]['A']}")
# #     print(f"1/f noise - alpha values: {noise_params[0]['alpha']}")
    
# # if noise_params[2]:  # If white noise params exist  
# #     print(f"White noise - C values: {noise_params[2]['C']}")
    
# # if noise_params[1]:  # If Lorentzian params exist
# #     print(f"Lorentzian noise - omega_0 values: {noise_params[1]['omega_0']}")
# #     print(f"Lorentzian noise - gamma values: {noise_params[1]['gamma']}")
# #     print(f"Lorentzian noise - A values: {noise_params[1]['A']}")
    
# # if noise_params[3]:  # If combined fraction params exist
# #     print(f"Combined fraction - A values: {noise_params[3]['A']}")
# #     print(f"Combined fraction - alpha values: {noise_params[3]['alpha']}")
# #     print(f"Combined fraction - omega_0 values: {noise_params[3]['omega_0']}")
# #     print(f"Combined fraction - gamma values: {noise_params[3]['gamma']}")

############################################################################

# Create save directory structure
subfolder_str_1 = f"tau_p{finite_width_params['tau_p']}/"
subfolder_str_2 = f"N1f{finite_width_params['N1f']}_Nlor{finite_width_params['Nlor']}_NC{finite_width_params['NC']}_Ndpl{finite_width_params['Ndpl']}/"

save_dir = global_filepath + subfolder_str_1 + subfolder_str_2

if combined_loss:
    save_dir += "combined_loss/"
if delta_approx:
    save_dir += "delta_approx/"
else:
    save_dir += "finite_width_FF/"

if not experimental_data:
    if noise:
        print(f"Gaussian σ: {noise_lvl}")
        save_dir += f"gaussian_noise_{noise_lvl}/"
    else:
        save_dir += "without_noise/"

save_dir += f"F{F}/"
save_dir += f"C_t_max{C_t_max:.4f}_min{C_t_min:.4f}/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Find all existing run directories
existing_runs = []
for item in os.listdir(save_dir):
    if item.startswith("run") and os.path.isdir(os.path.join(save_dir, item)):
        try:
            # Extract the number from "runX"
            run_num = int(item[3:])  # Remove "run" prefix
            existing_runs.append(run_num)
        except ValueError:
            # Skip if the suffix isn't a number
            continue

# Determine next run number
next_run = max(existing_runs) + 1 if existing_runs else 0

# Create the directory
save_dir = os.path.join(save_dir, f"run{next_run}/")
os.makedirs(save_dir, exist_ok=True)

######################################################################

for n in n_values:
    results_dict[n] = {}

    finite_width_params = {
        "N1f": N1f, # Number of 1/f noise parameters
        "Nlor": Nlor, # Number of Lorentzian noise parameters
        "NC": NC, # Number of white noise parameters
        "Ndpl": Ndpl, # Number of double power law noise parameters
        "N": n, #CPMG-N (test spin echo)
        "tau_p": tau_p, #pi pulse width in mircoseconds
        "integration_method": integration_method, # The method to use for integration. Options are "quad","trapezoid",and "simpson".
        "omega_resolution": omega_resolution, # 6*10**5 The number of points to use when integrating to yield the coherence decay profile, C(t), unless method is "quad", in which case, this is the resolution used to determine the filter function and integrand outputs.
        "omega_range" : omega_range, # The number of peaks to use in the filter function
        "num_peaks_cutoff": num_peaks_cutoff, # must be >=0 and <num_peaks. If this value is set, then only the top num_peaks_cutoff peaks will be used in the integration. This can be useful for speeding up the integration when the noise spectrum has many peaks.
        "peak_resolution": peak_resolution, # For every method except "quad", you can specify additional points around the singular points of the filter function. This argument specifies the number of points to use around each singular point.
        # "quad_limit": 10**5, # The maximum number of subdivisions in the adaptive quadrature algorithm
        # "epsabs": 1.49e-8, # Absolute tolerance for the quadrature integration
        # "epsrel": 1.49e-8, # Relative tolerance for the quadrature integration
    }

    results_dict[n]["F"] = F
    results_dict[n]["finite_width_params"] = finite_width_params
    results_dict[n]["true_params"] = noise_params


    if experimental_data:
        with open("data/example_XY8data_normtotpi1.json","r") as file:
            data = json.load(file)

        print(f"Using experimental data for n={n}")
        del data["y_fit"]
        df = pd.DataFrame(data)

        n_df = df[df["N_pi"]==n]
        t_points = (10**-3)*np.array(n_df["time_points"].to_list()).flatten()
        C_t_observed = np.array(n_df["C_t"].to_list()).flatten()
        t_points = t_points[find_widest_contiguous_stretch(C_t_observed, C_t_min, C_t_max)]
        C_t_observed = C_t_observed[find_widest_contiguous_stretch(C_t_observed, C_t_min, C_t_max)]

    else:
        # Find time range
        t_start, t_end = find_time_range_for_C_t_bounds(
            C_t_max, C_t_min,
            finite_width_params,
            ns.noise_spectrum_combination,
            noise_params,
            delta=delta_approx
        )

        if not delta_approx and t_start is not None:
            t_start = max(t_start, finite_width_params["N"] * finite_width_params["tau_p"] + 1e-10)

        time_resolution = 51 # Number of time points to use in the C(t) profile
        if t_start is not None and t_end is not None:
            t_points = np.logspace(np.log10(t_start), np.log10(t_end), time_resolution)
            print(f"N={n}: Time range: {t_start:.6f} to {t_end:.6f}")
        else:
            print(f"N={n}: Could not find both crossing points")
            t_points = np.logspace(np.log10(finite_width_params["N"]*finite_width_params["tau_p"]+1e-10), np.log10(300), time_resolution)

    results_dict[n]["t_points"] = t_points
    
    # Update omega range
    w_min = n*np.pi/(np.max(t_points))
    w_max = n*np.pi/(np.min(t_points))
    finite_width_params["omega_range"] = (10**np.floor(np.log10(w_min)), 
                                        np.max([10**(np.ceil(np.log10(w_max))),
                                                10**(np.floor(np.log10(w_min))+4)]))
    
    # Generate C_t_observed
    if experimental_data:
        results_dict[n]["C(t)_true"] = C_t_observed
        print(f"N={n}: C(t) Range:", (np.max(C_t_observed), np.min(C_t_observed)))
        if noise:
            C_t_dirty = C_t_observed.copy()
            results_dict[n]["C(t)_dirt"] = C_t_dirty
    else:
        C_t_observed = func_to_fit(t_points, ns.noise_spectrum_combination, *noise_params, delta=delta_approx, **finite_width_params)
        C_t_observed = F * C_t_observed 

        results_dict[n]["C(t)_true"] = C_t_observed
        print(f"N={n}: C(t) Range:", (np.max(C_t_observed), np.min(C_t_observed)))

        # Add noise if specified
        if noise:
            C_t_clean = C_t_observed.copy()
            C_t_observed, _ = add_gaussian_noise(C_t_observed, noise_level=noise_lvl, noise_type='absolute')
            C_t_dirty = C_t_observed.copy()
            results_dict[n]["C(t)_dirt"] = C_t_dirty
            print(f"N={n}: MSE Between true C(t) and noisy C(t):", mean_squared_error(C_t_clean, C_t_observed))

    # Store for combined fitting
    C_t_observed_dict[n] = C_t_observed
    times_dict[n] = t_points
    fixed_kwargs_dict[n] = finite_width_params

if combined_loss:
    # for Lorentzian noise, omega_0 is fixed at 0 for all n values since a non-zero value would be unphysical.
    bounds = finite_width_params["N1f"]*[(0,10**4),(0,10)] + finite_width_params["Nlor"]*[(0,10**4),(0, 0),(0,10**4)] + finite_width_params["NC"]*[(0,10)]+ finite_width_params["NC"]*[(0,10)] + finite_width_params["Ndpl"]*[(0,10**6),(0,10),(0, 10),(1e-10, 10**4)]
    print("\n=== COMBINED LOSS OPTIMIZATION ===")
    print("Fitting parameters to minimize combined loss across all n values...")
    
    # if local optimization, use multiple random initial guesses to avoid local minima
    if meth == "L-BFGS-B":
        r = 10 if not delta_approx else 50
    else:
        r = 1 

    best_combined_loss = np.inf
    best_combined_result = None
    all_combined_results = []  # Store all results for error analysis
    
    for k in range(r):
        if k == 0:
            # Use fixed initial guess for first run, so that you can customize if you want a specific starting point
            initial_args = [{'A': [10]*finite_width_params["N1f"], 'alpha': [1]*finite_width_params["N1f"]}, 
                           {"A": [1]*finite_width_params["Nlor"], 
                            "omega_0": [0]*finite_width_params["Nlor"], 
                            "gamma": [1]*finite_width_params["Nlor"]}, 
                           {'C': [0]*finite_width_params["NC"]},
                           {'A': [1]*finite_width_params["Ndpl"], 
                            'alpha': [1]*finite_width_params["Ndpl"], 
                            'beta': [2]*finite_width_params["Ndpl"], 
                            'gamma': [1]*finite_width_params["Ndpl"]}]
        else:
            # Generate alpha first
            alpha_vals = np.random.uniform(low=0, high=10, size=(finite_width_params["Ndpl"],))
            # Generate beta such that beta > alpha
            beta_vals = np.random.uniform(low=alpha_vals, high=10, size=(finite_width_params["Ndpl"],))

            initial_args = [{'A': list(np.random.uniform(low=1, high=10**12, size=(finite_width_params["N1f"],))),
                            'alpha': list(np.random.uniform(low=0, high=10, size=(finite_width_params["N1f"],)))},
                            {"A": list(np.random.uniform(low=1, high=10**8, size=(finite_width_params["Nlor"],))),
                                "omega_0": list(np.random.uniform(low=0, high=0, size=(finite_width_params["Nlor"],))),
                                "gamma": list(np.random.uniform(low=0, high=10**4, size=(finite_width_params["Nlor"],)))},
                            {'C': list(np.random.uniform(low=0, high=10, size=(finite_width_params["NC"],)))},
                            {'A': list(np.random.uniform(low=1, high=10**6, size=(finite_width_params["Ndpl"],))),
                                'alpha': list(alpha_vals),
                                'beta': list(beta_vals),
                                'gamma': list(np.random.uniform(low=1, high=10**4, size=(finite_width_params["Ndpl"],)))}]

        # Run combined optimization
        # We have pre-selected values for population and iterations based on trial and error to balance speed and accuracy. You can adjust these if you want.
        optimized_args, optimized_errors, opt_result = fit_coherence_decay_combined(
            C_t_observed_dict, times_dict, ns.noise_spectrum_combination, initial_args, 
            fixed_kwargs_dict, n_values, bounds, method=meth, delta=delta_approx, 
            loss_type=loss_type, noise_level=noise_lvl, population=int(sys.argv[10]) if len(sys.argv) > 10 else 10, iterations=int(sys.argv[9]) if len(sys.argv) > 9 else 50
        )

        # Store all results for error analysis
        all_combined_results.append((optimized_args, optimized_errors, opt_result))

        if opt_result.fun < best_combined_loss:
            best_combined_loss = opt_result.fun
            best_combined_result = (optimized_args, optimized_errors, opt_result)
            print(f"New best combined loss: {best_combined_loss} at iteration {k+1}")
            print(f"Combined optimized parameters: {optimized_args}")
    
    # Calculate parameter statistics if r > 1
    if r > 1:
        print("\n=== PARAMETER STATISTICS FROM MULTIPLE RUNS ===")
        
        # Extract parameter values from all runs
        param_values = {'A': [], 'alpha': [], 'C': []}
        
        for optimized_args, _, _ in all_combined_results:
            # Extract A values
            if optimized_args[0].get('A'):
                param_values['A'].extend(optimized_args[0]['A'])
            
            # Extract alpha values
            if optimized_args[0].get('alpha'):
                param_values['alpha'].extend(optimized_args[0]['alpha'])
            
            # Extract C values
            if len(optimized_args) > 2 and optimized_args[2].get('C'):
                param_values['C'].extend(optimized_args[2]['C'])
        
        # Calculate statistics
        param_stats = {}
        for param_name, values in param_values.items():
            if values:  # Only calculate if we have values
                param_stats[param_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'values': values
                }
                print(f"{param_name}: mean={param_stats[param_name]['mean']:.4f} ± {param_stats[param_name]['std']:.4f}")
                print(f"  Range: [{param_stats[param_name]['min']:.4f}, {param_stats[param_name]['max']:.4f}]")
                print(f"  Median: {param_stats[param_name]['median']:.4f}")
        
        # Store parameter statistics
        combined_param_stats = param_stats
    else:
        combined_param_stats = None
    
    # Store the best result for all n values
    for n in n_values:
        results_dict[n]["optimized_args"] = best_combined_result[0]
        results_dict[n]["optimized_errors"] = best_combined_result[1]
        results_dict[n]["opt_result"] = best_combined_result[2]
        results_dict[n]["all_optimization_results"] = all_combined_results
        results_dict[n]["parameter_statistics"] = combined_param_stats
    
    print(f"\nFinal combined loss: {best_combined_loss}")
    print(f"Final combined parameters: {best_combined_result[0]}")

else:
    print("\n=== INDIVIDUAL OPTIMIZATION ===")
    print("Fitting parameters separately for each n value...")
    
    # Original individual fitting logic
    for n in n_values:

        # *********** NOTE: BOUNDS MAY NEED TO BE ADJUSTED BASED ON n ***********
        # Define bounds. Empirically chosen based on n. The smaller the bounds, the faster the optimization due to smaller search space. 
        # These can be adjusted based on expected parameter ranges.
        if n >= 128:
            alpha_max = 10
            A_max = 10**10
        else:
            A_max = 10000
            alpha_max = 5
        # for Lorentzian noise, omega_0 is fixed at 0 for all n values since a non-zero value would be unphysical.
        bounds = finite_width_params["N1f"]*[(0,A_max),(0,alpha_max)] + finite_width_params["Nlor"]*[(0,10**5),(0, 0),(0,10**4)] + finite_width_params["NC"]*[(0,10)] + finite_width_params["Ndpl"]*[(0,10**6),(0,10),(0, 10),(1e-10, 10**4)]

        print(f"\nOptimizing for N={n}...")
        MSE = np.inf
        
        finite_width_params = fixed_kwargs_dict[n]
        t_points = times_dict[n]
        C_t_observed = C_t_observed_dict[n]
        
        # if local optimization, use multiple random initial guesses to avoid local minima
        if meth == "L-BFGS-B":
            r = 10 if not delta_approx else 10**4
        else:
            r = 1

        all_individual_results = []  # Store all results for error analysis
        
        for k in range(r):
            if k == 0:
                # Use fixed initial guess for first run, so that you can customize if you want a specific starting point
                initial_args = [{'A': [10]*finite_width_params["N1f"], 'alpha': [1]*finite_width_params["N1f"]}, 
                            {"A": [0]*finite_width_params["Nlor"], 
                                "omega_0": [0]*finite_width_params["Nlor"], 
                                "gamma": [0]*finite_width_params["Nlor"]}, 
                            {'C': [0]*finite_width_params["NC"]},
                            {'A': [0]*finite_width_params["Ndpl"], 
                                'alpha': [1]*finite_width_params["Ndpl"], 
                                'beta': [2]*finite_width_params["Ndpl"], 
                                'gamma': [1]*finite_width_params["Ndpl"]}]
            else:
                # Generate alpha first
                alpha_vals = np.random.uniform(low=0, high=3, size=(finite_width_params["Ndpl"],))
                # Generate beta such that beta > alpha
                beta_vals = np.random.uniform(low=alpha_vals, high=3, size=(finite_width_params["Ndpl"],))

                initial_args = [{'A': list(np.random.uniform(low=1, high=10**12, size=(finite_width_params["N1f"],))),
                                'alpha': list(np.random.uniform(low=0, high=10, size=(finite_width_params["N1f"],)))},
                                {"A": list(np.random.uniform(low=1, high=10**8, size=(finite_width_params["Nlor"],))),
                                    "omega_0": list(np.random.uniform(low=0, high=0, size=(finite_width_params["Nlor"],))),
                                    "gamma": list(np.random.uniform(low=1, high=10**4, size=(finite_width_params["Nlor"],)))},
                                {'C': list(np.random.uniform(low=0, high=10, size=(finite_width_params["NC"],)))},
                                {'A': list(np.random.uniform(low=1, high=10**6, size=(finite_width_params["Ndpl"],))),
                                    'alpha': list(alpha_vals),
                                    'beta': list(beta_vals),
                                    'gamma': list(np.random.uniform(low=1, high=10**4, size=(finite_width_params["Ndpl"],)))}]

            # Run individual optimization
            # We have pre-selected values for population and iterations based on trial and error to balance speed and accuracy. You can adjust these if you want.
            optimized_args, optimized_errors, opt_result = fit_coherence_decay(
                C_t_observed, t_points, ns.noise_spectrum_combination, initial_args, 
                finite_width_params, bounds, method=meth, delta=delta_approx, 
                loss_type=loss_type, noise_level=noise_lvl, population=int(sys.argv[10]) if len(sys.argv) > 10 else 10, iterations=int(sys.argv[9]) if len(sys.argv) > 9 else 50
            )

            # Store all results for error analysis
            all_individual_results.append((optimized_args, optimized_errors, opt_result))

            if opt_result.fun < MSE:
                MSE = opt_result.fun
                print(f"N={n}: New best MSE: {MSE} at iteration {k+1}")
                results_dict[n]["optimized_args"] = optimized_args
                results_dict[n]["optimized_errors"] = optimized_errors
                results_dict[n]["opt_result"] = opt_result
        
        # Calculate parameter statistics if r > 1
        if r > 1:
            print(f"\n=== PARAMETER STATISTICS FOR N={n} FROM MULTIPLE RUNS ===")
            
            # Extract parameter values from all runs
            param_values = {'A': [], 'alpha': [], 'C': []}
            
            for optimized_args, _, _ in all_individual_results:
                # Extract A values
                if optimized_args[0].get('A'):
                    param_values['A'].extend(optimized_args[0]['A'])
                
                # Extract alpha values
                if optimized_args[0].get('alpha'):
                    param_values['alpha'].extend(optimized_args[0]['alpha'])
                
                # Extract C values
                if len(optimized_args) > 2 and optimized_args[2].get('C'):
                    param_values['C'].extend(optimized_args[2]['C'])
            
            # Calculate statistics
            param_stats = {}
            for param_name, values in param_values.items():
                if values:  # Only calculate if we have values
                    param_stats[param_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'values': values
                    }
                    print(f"{param_name}: mean={param_stats[param_name]['mean']:.4f} ± {param_stats[param_name]['std']:.4f}")
                    print(f"  Range: [{param_stats[param_name]['min']:.4f}, {param_stats[param_name]['max']:.4f}]")
                    print(f"  Median: {param_stats[param_name]['median']:.4f}")
            
            # Store parameter statistics
            results_dict[n]["parameter_statistics"] = param_stats
        else:
            results_dict[n]["parameter_statistics"] = None
        
        # Store all optimization results
        results_dict[n]["all_optimization_results"] = all_individual_results

#####################################################################################

# Continue with plotting and saving logic for each n value
for n in n_values:
    print(f"\nProcessing plots for N={n}...")
    
    finite_width_params = fixed_kwargs_dict[n]
    t_points = times_dict[n]
    C_t_observed = C_t_observed_dict[n]

    # Generate the plotting data
    green_line = cp.noise_inversion_delta(t_points, C_t_observed)
    
    omega_values = n*np.pi/t_points
    omega_values = omega_values[(green_line != np.inf) & (green_line != -np.inf)]
    time_points = t_points[(green_line != np.inf) & (green_line != -np.inf)]
    green_line = green_line[(green_line != np.inf) & (green_line != -np.inf)]

    fitted_noise_spectrum = ns.noise_spectrum_combination(omega_values, *results_dict[n]["optimized_args"])
    S_w_synthetic = ns.noise_spectrum_combination(omega_values, *noise_params)
    
    C_t_plot = F*func_to_fit(t_points, ns.noise_spectrum_combination, *noise_params, delta=delta_approx, **finite_width_params)

    # PLOT 1: Noise Spectrum Comparison
    plt.figure(figsize=(10, 6))
    if not experimental_data and len(omega_values) > 0:
        plt.plot(omega_values, S_w_synthetic, label=f"Original Noise, ({format_parameters(noise_params)})", marker='o')

    if len(omega_values) > 0:
        plt.plot(omega_values, fitted_noise_spectrum, label=f"Fitted Noise ({format_parameters(results_dict[n]['optimized_args'])})", linestyle='--')
    else:
        pass

    # try:
    #     C_t1 = 0.95
    #     C_t2 = 0.05
    #     k = np.where((C_t_plot < C_t1) & (C_t_plot > C_t2))[0]

    #     w_1 = n*np.pi/np.max(t_points[k])
    #     w_2 = n*np.pi/np.min(t_points[k])

    #     # Add vertical red lines
    #     plt.axvline(w_1, color='red', linestyle='-', linewidth=2, label=f'{C_t1:.3f} > C(t) > {C_t2:.3f}')
    #     plt.axvline(w_2, color='red', linestyle='-', linewidth=2)
    # except Exception as e:
    #     print(f"Error in calculating vertical lines: {e}")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('ω (rad/μs)')
    plt.ylabel('S(ω)')
    plt.legend()

    if not experimental_data:
        if delta_approx:
            title = f"Delta approx S(ω), n = {n} \n {format_parameters(noise_params)}"
        else:
            title = f"S(ω), n = {n} \n {format_parameters(noise_params)}"
    else:
        if delta_approx:
            title = f"Delta approx S(ω), n = {n}"
        else:
            title = f"S(ω), n = {n}"
    
    if combined_loss:
        title += " [Combined Loss]"
    
    plt.title(title)
    try:
        plt.savefig(save_dir + title + ".png", dpi="figure", bbox_inches="tight")
    except Exception as e:
        print(f"Error saving plot: {e}")
        print(save_dir)
        plt.savefig(save_dir + "S(ω)" + ".png", dpi="figure", bbox_inches="tight")
    # plt.show()
    plt.close()

    # PLOT 2: Coherence Decay Comparison
    plt.figure(figsize=(10, 6))
    if noise:
        plt.plot(t_points, results_dict[n]["C(t)_dirt"], label=f"Noisy C(t)", marker='o')
        plt.plot(t_points, results_dict[n]["C(t)_true"], label=f"Original C(t)", linestyle='--')
    else:
        try:
            if not experimental_data:
                plt.plot(t_points, C_t_plot, label=f"Original C(t)")
            fitted_C_t = func_to_fit(t_points, ns.noise_spectrum_combination, *results_dict[n]["optimized_args"], delta=delta_approx, **finite_width_params)
        except Exception as e:
            if not experimental_data:
                plt.plot(time_points, C_t_plot, label=f"Original C(t)")
            fitted_C_t = func_to_fit(time_points, ns.noise_spectrum_combination, *results_dict[n]["optimized_args"], delta=delta_approx, **finite_width_params)
    
    results_dict[n]["C(t)_fitted"] = fitted_C_t
    plt.plot(t_points, fitted_C_t, label=f"Fitted C(t) ({format_parameters(results_dict[n]['optimized_args'])})", linestyle='--')

    if delta_approx:
        plt.axvline(finite_width_params["N"]*finite_width_params["tau_p"], color='red', linestyle='-', linewidth=2, label=f't = N*τ_p = {finite_width_params["N"]*finite_width_params["tau_p"]:.3f} µs')
    
    plt.xscale('log')
    plt.xlabel('Time (µs)')
    plt.ylabel('C(t)')
    plt.legend()

    if not experimental_data:
        if delta_approx:
            title = f"Delta approx C(t), n = {n} \n {format_parameters(noise_params)}"
        else:
            title = f"C(t), n = {n} \n {format_parameters(noise_params)}"
    else:
        if delta_approx:
            title = f"Delta approx C(t), n = {n}"
        else:
            title = f"C(t), n = {n}"
    
    if combined_loss:
        title += " [Combined Loss]"
    
    plt.title(title)
    try:
        plt.savefig(save_dir + title + ".png", dpi="figure", bbox_inches="tight")
    except Exception as e:
        print(f"Error saving plot: {e}")
        # Fallback save if the title is too long or contains invalid characters
        plt.savefig(save_dir + "C(t)" + ".png", dpi="figure", bbox_inches="tight")
    # plt.show()
    plt.close()

    if not experimental_data:
        # PLOT 3: Chi(t) Analysis
        plt.figure(figsize=(10, 6))

        chi_t = -np.log(C_t_plot)
        chi_max = np.log(1/0.05)
        chi_min = np.log(1/0.95)

        try:
            if len(t_points[(chi_t != np.inf) & (chi_t >= 10**-2)]) > 0:
                plt.plot(t_points[(chi_t != np.inf) & (chi_t >= 10**-2)], chi_t[(chi_t != np.inf) & (chi_t >= 10**-2)], label=f"Original χ(t)")
        except Exception as e:
            plt.plot(time_points[(chi_t != np.inf) & (chi_t >= 10**-2)], chi_t[(chi_t != np.inf) & (chi_t >= 10**-2)], label=f"Original χ(t)")

        plt.axhline(y=chi_max, color='red', linestyle='--', label=f'{chi_max:.3f} > χ(t) > {chi_min:.3f}')
        plt.axhline(y=chi_min, color='red', linestyle='--')
        
        try:
            guideline_t = t_points[(chi_t != np.inf) & (chi_t >= 10**-2)]
            # guideline_t_alpha = guideline_t**(results_dict[n]['optimized_args'][0]['alpha'][0]+1)
            # plt.plot(t_points[(chi_t != np.inf) & (chi_t >= 10**-2)], guideline_t_alpha*(np.mean(chi_t)/np.mean(guideline_t_alpha)), label=r"$t^{\alpha+1}$ Guideline")
            if len(t_points[(chi_t != np.inf) & (chi_t >= 10**-2)]) > 0:
                plt.plot(t_points[(chi_t != np.inf) & (chi_t >= 10**-2)], guideline_t*(np.mean(chi_t)/np.mean(guideline_t)), label=r"$t$ Guideline")
        except Exception as e:
            guideline_t = time_points[(chi_t != np.inf) & (chi_t >= 10**-2)]
            # guideline_t_alpha = guideline_t**(results_dict[n]['optimized_args'][0]['alpha'][0]+1)
            # plt.plot(time_points[(chi_t != np.inf) & (chi_t >= 10**-2)], guideline_t_alpha*(np.mean(chi_t)/np.mean(guideline_t_alpha)), label=r"$t^{\alpha+1}$ Guideline")
            plt.plot(time_points[(chi_t != np.inf) & (chi_t >= 10**-2)], guideline_t*(np.mean(chi_t)/np.mean(guideline_t)), label=r"$t$ Guideline")
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Time (µs)")
        plt.ylabel("χ(t)")
        plt.legend()

        if not experimental_data:
            if delta_approx:
                title = f"Delta approx χ(t), n = {n} \n {format_parameters(noise_params)}"
            else:
                title = f"χ(t), n = {n} \n {format_parameters(noise_params)}"
        else:
            if delta_approx:
                title = f"Delta approx χ(t), n = {n}"
            else:
                title = f"χ(t), n = {n}"

        if combined_loss:
            title += " [Combined Loss]"

        plt.title(title)
        try:
            plt.savefig(save_dir + title + ".png", dpi="figure", bbox_inches="tight")
        except Exception as e:
            print(f"Error saving plot: {e}")
            # Fallback save if the title is too long or contains invalid characters
            plt.savefig(save_dir + "χ(t)" + ".png", dpi="figure", bbox_inches="tight")

        # plt.show()
        plt.close()

    # Save numpy arrays if noise was added
    if noise:
        if not experimental_data:
            np.save(save_dir + f"synthetic_C_t_dirty_n{n}.npy", results_dict[n]["C(t)_dirt"])
        else:
            np.save(save_dir + f"dirty_C_t_n{n}.npy", results_dict[n]["C(t)_dirt"])

# Save results
with open(save_dir + f"results.pkl", 'wb') as f:
    pickle.dump(results_dict, f)

print(f"\nResults saved to: {save_dir}")
if combined_loss:
    print("Combined loss optimization completed!")
else:
    print("Individual optimization completed!")

print("\n" + "="*50)
print("Creating Combined Analysis Plot")
print("="*50)

# Create the combined analysis plot using your existing variables
fig = create_combined_analysis_plot(
    results_dict=results_dict,
    n_values=n_values,
    combined_loss=combined_loss,
    save_dir=save_dir,
    experimental_data=experimental_data,
    noise_params=noise_params if not experimental_data else None,
    delta_approx=delta_approx,
    F=F,
    include_inversion = False)