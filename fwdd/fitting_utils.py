import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sklearn.metrics import mean_squared_error

from .noise_spectra import noise_spectrum_combination
from .coherence_profile import noise_inversion_delta
from .noise_learning_fitting import func_to_fit

# Bootstrapping functionality
import random
from typing import Dict, List, Tuple, Set


def add_gaussian_noise(signal, noise_level=0.01, noise_type='absolute', seed=None):
    """
    Add Gaussian noise to an input vector.
    
    Parameters:
    -----------
    signal : array-like
        Input signal/vector to add noise to
    noise_level : float, default=0.01
        Noise strength parameter
        - For 'relative': fraction of signal range (0.01 = 1% of signal range)
        - For 'absolute': standard deviation of noise in original units
        - For 'snr': signal-to-noise ratio in dB
    noise_type : str, default='relative'
        Type of noise scaling:
        - 'relative': noise proportional to signal range
        - 'absolute': fixed noise standard deviation
        - 'snr': noise based on signal-to-noise ratio
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    noisy_signal : ndarray
        Signal with added Gaussian noise
    noise : ndarray
        The noise that was added (for analysis)
        
    Examples:
    ---------
    >>> signal = np.sin(np.linspace(0, 2*np.pi, 100))
    >>> noisy_signal, noise = add_gaussian_noise(signal, noise_level=0.05)
    >>> 
    >>> # Different noise types
    >>> noisy_rel, _ = add_gaussian_noise(signal, 0.02, 'relative')  # 2% of range
    >>> noisy_abs, _ = add_gaussian_noise(signal, 0.1, 'absolute')   # σ = 0.1
    >>> noisy_snr, _ = add_gaussian_noise(signal, 20, 'snr')         # 20 dB SNR
    """
    
    # Convert to numpy array
    signal = np.asarray(signal)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate noise standard deviation based on type
    if noise_type == 'relative':
        # Noise as fraction of signal range
        signal_range = np.max(signal) - np.min(signal)
        noise_std = noise_level * signal_range
        
    elif noise_type == 'absolute':
        # Direct standard deviation
        noise_std = noise_level
        
    elif noise_type == 'snr':
        # Signal-to-noise ratio in dB
        signal_power = np.mean(signal**2)
        snr_linear = 10**(noise_level / 10)  # Convert dB to linear
        noise_std = np.sqrt(signal_power / snr_linear)
        
    else:
        raise ValueError("noise_type must be 'relative', 'absolute', or 'snr'")
    
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, signal.shape)
    
    # Add noise to signal
    noisy_signal = signal + noise
    
    return noisy_signal, noise


def find_C_t_crossing_points(C_t_max, C_t_min, t_min=0, t_max=300, num_scan_points=1000, 
                           noise_spectrum=noise_spectrum_combination, noise_params=None, delta=False, **finite_width_params):
    """
    Find crossing points for both C_t_max and C_t_min thresholds. I.e.  the times where C(t) crosses C_t_max and C_t_min.

    Inputs:
    -------
    C_t_max : float
        0 < C_t_max < 1, Upper threshold for C(t)
    C_t_min : float
        0 < C_t_min < 1, Lower threshold for C(t)
    t_min : float, optional
        Minimum time to start searching (default 0)    
    t_max : float, optional
        Maximum time to search (default 300 µs)
    num_scan_points : int, optional
        Number of points to scan between t_min and t_max (default 1000)
    noise_spectrum : function
        Function to compute noise spectrum S(w) (default noise_spectrum_combination)
    noise_params : tuple or list, optional
        Parameters to pass to noise_spectrum function (will be unpacked with *)
    delta : bool, optional
        If True, use delta function case in func_to_fit
    finite_width_params : dict
        Parameters for finite-width pulse sequence (e.g. N, tau_p, etc.)

    Returns:
    --------
    dict with keys:
        'leftmost_C_t_max': leftmost point where C_t = C_t_max
        'rightmost_C_t_min': rightmost point where C_t = C_t_min
    """
    
    # Ensure all parameters are proper numeric types
    C_t_max = float(C_t_max)
    C_t_min = float(C_t_min)
    t_min = float(t_min)
    t_max = float(t_max)
    num_scan_points = int(num_scan_points)
    
    # Ensure noise_params is iterable, default to empty tuple if None
    if noise_params is None:
        noise_params = ()
    
    def find_leftmost_crossing(threshold, t_min, t_max, num_scan_points):
        """Find the leftmost (earliest) crossing point"""
        # Ensure all inputs to linspace are floats
        t_scan = np.linspace(float(t_min), float(t_max), int(num_scan_points))  # Forward order
        C_t_scan = func_to_fit(t_scan, noise_spectrum, *noise_params, delta=delta, **finite_width_params)
        
        diff_from_threshold = np.array(C_t_scan) - float(threshold)
        
        # Find first crossing from above to below (going forward in time)
        for i in range(len(diff_from_threshold) - 1):
            if diff_from_threshold[i] > 0 and diff_from_threshold[i+1] < 0:
                try:
                    def target_func(t):
                        C_t = func_to_fit([float(t)], noise_spectrum, *noise_params, delta=delta, **finite_width_params)
                        return C_t[0] - float(threshold)
                    
                    root = brentq(target_func, float(t_scan[i]), float(t_scan[i+1]))
                    return float(root)
                except ValueError:
                    continue
        return None
    
    def find_rightmost_crossing(threshold, t_min, t_max, num_scan_points):
        """Find the rightmost (latest) crossing point"""
        # Ensure all inputs to linspace are floats
        t_scan = np.linspace(float(t_max), float(t_min), int(num_scan_points))  # Reverse order
        C_t_scan = func_to_fit(t_scan, noise_spectrum, *noise_params, delta=delta, **finite_width_params)
        
        diff_from_threshold = np.array(C_t_scan) - float(threshold)
        
        # Find first crossing from below to above (going backwards in time)
        for i in range(len(diff_from_threshold) - 1):
            if diff_from_threshold[i] < 0 and diff_from_threshold[i+1] > 0:
                try:
                    def target_func(t):
                        C_t = func_to_fit([float(t)], noise_spectrum, *noise_params, delta=delta, **finite_width_params)
                        return C_t[0] - float(threshold)
                    
                    root = brentq(target_func, float(t_scan[i+1]), float(t_scan[i]))
                    return float(root)
                except ValueError:
                    continue
        return None
    
    # Find both crossing points
    leftmost_C_t_max = find_leftmost_crossing(C_t_max, t_min, t_max, num_scan_points)
    rightmost_C_t_min = find_rightmost_crossing(C_t_min, t_min, t_max, num_scan_points)
    
    return {
        'leftmost_C_t_max': leftmost_C_t_max,
        'rightmost_C_t_min': rightmost_C_t_min
    }

def find_time_range_for_C_t_bounds(C_t_max, C_t_min, finite_width_params, noise_spectrum, noise_params=None, delta=False):
    """
    Find the time range between leftmost C_t_max crossing and rightmost C_t_min crossing
    
    Inputs:
    -------
    C_t_max : float
        0 < C_t_max < 1, Upper threshold for C(t)
    C_t_min : float
        0 < C_t_min < 1, Lower threshold for C(t)
    finite_width_params : dict
        Parameters for finite-width pulse sequence (e.g. N, tau_p, etc.)
    noise_spectrum : function
        Function to compute noise spectrum S(w) (default noise_spectrum_combination)
    noise_params : tuple or list, optional
        Parameters to pass to noise_spectrum function (will be unpacked with *)
    delta : bool, optional
        If True, use delta function case in func_to_fit (default False)
    """

    t_max = 300
    num_scan_points = 1000

    if delta:
        t_min_initial = float(1e-10)  # Ensure it's a float
        t_max_search = float(t_max)     # Ensure it's a float
        loop_count = 0
    else:
        t_min_initial = float(finite_width_params["N"] * finite_width_params["tau_p"] + 1e-10)  # Ensure it's a float
        t_max_search = float(t_max)     # Ensure it's a float
        loop_count = 0
    
    while True:
        results = find_C_t_crossing_points(
            float(C_t_max),      # Ensure it's a float
            float(C_t_min),      # Ensure it's a float
            t_min_initial, 
            t_max_search, 
            num_scan_points,
            noise_spectrum,
            noise_params,
            delta=delta,
            **finite_width_params
        )
        
        leftmost_t = results['leftmost_C_t_max']
        rightmost_t = results['rightmost_C_t_min']
        
        # Check if we found both points
        if leftmost_t is not None and rightmost_t is not None:
            return float(leftmost_t), float(rightmost_t)  # Ensure return values are floats
        
        # If we didn't find both points, expand the search range
        if leftmost_t is None or rightmost_t is None:
            t_max_search *= 2  # Double the maximum time
            
            # Safety check to prevent infinite loops
            if loop_count > 100:
                print("Warning: Maximum iterations reached")
                print(f"Found leftmost C_t_max crossing: {leftmost_t}")
                print(f"Found rightmost C_t_min crossing: {rightmost_t}")
                break
            loop_count += 1
        else:
            break
    
    # Return None values as floats if no solution found
    return (float(leftmost_t) if leftmost_t is not None else None, 
            float(rightmost_t) if rightmost_t is not None else None)


def analyze_intervals(intervals, print_results=True):
    """
    Analyze overlapping intervals and return list of new intervals with their coverage info.
    
    Parameters:
    intervals: numpy array where each row is [start, end] of an interval
    print_results: bool, whether to print the analysis (default True)
    
    Returns:
    list of tuples: (start, end, containing_intervals_list)
    """
    # Sort intervals by start point (should already be sorted in your case)
    intervals = intervals[intervals[:, 0].argsort()]
    
    # Collect all boundary points
    all_points = set()
    for start, end in intervals:
        all_points.add(start)
        all_points.add(end)
    
    # Sort all boundary points
    boundary_points = sorted(all_points)
    
    if print_results:
        print("Interval Analysis:")
        print("=" * 50)
    
    result_intervals = []
    
    # For each segment between consecutive boundary points
    for i in range(len(boundary_points) - 1):
        segment_start = boundary_points[i]
        segment_end = boundary_points[i + 1]
        segment_mid = (segment_start + segment_end) / 2  # Test point in the middle
        
        # Find which intervals contain this segment
        containing_intervals = []
        for idx, (start, end) in enumerate(intervals):
            if start <= segment_mid <= end:
                containing_intervals.append(idx)
        
        # Skip empty segments
        if len(containing_intervals) == 0:
            continue
            
        # Add to results
        result_intervals.append((segment_start, segment_end, containing_intervals))
        
        # Print the segment information
        if print_results:
            if len(containing_intervals) == 1:
                interval_idx = containing_intervals[0]
                print(f"Interval {interval_idx} only: [{segment_start:.3f}, {segment_end:.3f}]")
            else:
                interval_names = " & ".join([f"{idx}" for idx in containing_intervals])
                print(f"Intervals {interval_names} overlap: [{segment_start:.3f}, {segment_end:.3f}]")
    
    return result_intervals

def find_widest_contiguous_stretch(arr, lb, ub):
   '''
   Find the longest contiguous stretch of values in arr that lie between lb and ub (inclusive).
   Inputs:
    - arr: 1D numpy array of numeric values
    - lb: lower bound (inclusive)
    - ub: upper bound (inclusive)
    Outputs:
    - List of indices corresponding to the longest contiguous stretch within [lb, ub]
    1. Create a boolean mask where values are within [lb, ub].
    2. Identify start and end indices of contiguous stretches.
    3. Find the longest stretch and return its indices.
    4. If no values are in range, return an empty list.
    5. Handles edge cases where the longest stretch is at the start or end of the array.
   '''
   # Create boolean mask for values between lb and ub
   in_range_mask = (arr >= lb) & (arr <= ub)
   
   if not np.any(in_range_mask):
       return []  # No values in range
   
   # Find start and end of each contiguous stretch
   # Add padding to handle edge cases
   padded_mask = np.concatenate([[False], in_range_mask, [False]])
   
   # Find where stretches start and end
   diff = np.diff(padded_mask.astype(int))
   starts = np.where(diff == 1)[0]  # Where False->True
   ends = np.where(diff == -1)[0] - 1  # Where True->False, adjust for padding
   
   # Find the longest stretch
   lengths = ends - starts + 1
   longest_idx = np.argmax(lengths)
   
   # Return indices of the longest stretch
   start_idx = starts[longest_idx]
   end_idx = ends[longest_idx]
   
   return list(range(start_idx, end_idx + 1))

def format_parameters(args_list, break_lines=True):
    """Format parameter list for legend display"""
    param_str = ""

    non_empty_mask = [bool(d) for d in args_list]
    last_index = np.where(non_empty_mask)[0][-1] if any(non_empty_mask) else -1

    # 1/f noise parameters
    if args_list[0] and 'A' in args_list[0] and args_list[0]['A']:
        A_vals = args_list[0]['A']
        alpha_vals = args_list[0]['alpha']
        if len(A_vals) == 1:
            param_str += f"A={A_vals[0]:.2f}, α={alpha_vals[0]:.2f}"
        else:
            param_str += f"A={A_vals}, α={alpha_vals}"

        if last_index != 0 and break_lines:
            param_str += "\n"
    
    # Lorentzian parameters (if present)
    if len(args_list) > 1 and args_list[1]:
        lor_params = []
        for key in ['omega_0', 'gamma', 'A']:
            if key in args_list[1] and args_list[1][key]:
                vals = args_list[1][key]
                if len(vals) == 1:
                    if key == "A":
                        lor_params.append(f"A={vals[0]:.2f}")
                    else:
                        lor_params.append(rf"$\{key}$={vals[0]:.2f}")
                else:
                    if key == "A":
                        lor_params.append(f"A={vals}")
                    else:
                        lor_params.append(rf"$\{key}$={vals}")

        if lor_params:
            param_str += ", ".join(lor_params)

        if last_index != 1 and break_lines:
            param_str += "\n"

    # White noise parameters
    if len(args_list) > 2 and args_list[2] and 'C' in args_list[2] and args_list[2]['C']:
        C_vals = args_list[2]['C']

        if len(C_vals) == 1:
            param_str += f"C={C_vals[0]:.2f}"
        else:
            param_str += f"C={C_vals}"

        if last_index != 2 and break_lines:
            param_str += "\n"

    # Fourth dictionary parameters (A, omega_0, gamma, alpha)
    if len(args_list) > 3 and args_list[3]:
        fourth_params = []
        for key in ['A', 'alpha', 'beta', 'gamma']:
            if key in args_list[3] and args_list[3][key]:
                vals = args_list[3][key]
                if len(vals) == 1:
                    if key == "A":
                        fourth_params.append(f"A={vals[0]:.2f}")
                    else:
                        fourth_params.append(rf"$\{key}$={vals[0]:.2f}")
                else:
                    if key == "A":
                        fourth_params.append(f"A={vals}")
                    else:
                        fourth_params.append(rf"$\{key}$={vals}")

        if fourth_params:
            param_str += ", ".join(fourth_params)

        if last_index != 3 and break_lines:
            param_str += "\n"
    
    return param_str

def create_combined_analysis_plot(results_dict, n_values, combined_loss=False, save_dir=None, 
                                 experimental_data=True, noise_params=None, delta_approx=False, F=1, include_inversion=False):
    """
    Create a combined plot showing both coherence decay profiles and noise spectra.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results for each n value
    n_values : list
        List of n values to plot
    combined_loss : bool
        Whether combined loss optimization was used
    save_dir : str
        Directory to save the plot
    experimental_data : bool
        Whether using experimental data
    noise_params : list
        Original noise parameters (for synthetic data)
    delta_approx : bool
        Whether delta approximation was used
    F : float
        Scaling factor
    """
    
    # Color scheme for consistent plotting
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    
    # Create figure with subplots - increased width to accommodate external legends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: C(t) data and fits for all N values
    ax1.set_title("Coherence Decay Profile" + (" - Combined Fit" if combined_loss else ""), fontsize=14)
    ax1.set_xlabel("Time (μs)", fontsize=12)
    ax1.set_ylabel("C(t)", fontsize=12)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Collect omega bounds for global range
    omega_bounds = []
    
    # Plot data and fits for each N value
    for i, n in enumerate(n_values):
        color = colors[i % len(colors)]
        
        # Get data from results
        t_points = results_dict[n]["t_points"]
        finite_width_params = results_dict[n]["finite_width_params"]
        optimized_args = results_dict[n]["optimized_args"]
        
        # Get observed data
        if "C(t)_dirt" in results_dict[n]:
            C_t_observed = results_dict[n]["C(t)_dirt"]
            C_t_clean = results_dict[n]["C(t)_true"]
            # Plot both clean and noisy data
            ax1.plot(t_points, C_t_clean, color=color, linewidth=1, alpha=0.7,
                    label=f"Clean Data N={n}")
            ax1.scatter(t_points, C_t_observed, color=color, alpha=0.8, s=20,
                       label=f"Noisy Data N={n}", marker='o', edgecolors='black', linewidths=0.5)
        else:
            C_t_observed = results_dict[n]["C(t)_true"]
            ax1.scatter(t_points, C_t_observed, color=color, alpha=0.8, s=30,
                       label=f"Data N={n}", marker='o', edgecolors='black', linewidths=0.5)

        t_points_special = np.concatenate((np.logspace(np.log10(n*finite_width_params["tau_p"]+1e-10),np.log10(t_points[0]),10),t_points))  # Extend for fitting
        # Calculate fitted C(t) using optimized parameters
        C_t_fitted = func_to_fit(t_points, noise_spectrum_combination, 
                                *optimized_args, delta=delta_approx, **finite_width_params)

        C_t_special = func_to_fit(t_points_special, noise_spectrum_combination, 
                                   *optimized_args, delta=delta_approx, **finite_width_params)
        
        # Scale if needed
        if not experimental_data:
            C_t_fitted = F * C_t_fitted
            C_t_special = F * C_t_special

        # Plot fitted curve
        ax1.plot(t_points_special, C_t_special, color=color, linewidth=2.5,
                label=f"Fit N={n}")
        
        # Calculate and display MSE for this N
        mse_n = mean_squared_error(C_t_observed, C_t_fitted)
        print(f"MSE for N={n}: {mse_n:.6f}")
        
        # Store omega bounds for global range
        omega_min = n * np.pi / np.max(t_points)
        omega_max = n * np.pi / np.min(t_points)
        omega_bounds.extend([omega_min, omega_max])
    
    # Place legend outside plot on the right
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Plot 2: Noise spectrum
    ax2.set_title("Spectral Noise Density" + (" - Combined Fit" if combined_loss else ""), fontsize=14)
    ax2.set_xlabel("ω (rad/μs)", fontsize=12)
    ax2.set_ylabel("S(ω)", fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Set global omega range
    omega_range_global = (np.min(omega_bounds), np.max(omega_bounds))
    omega_plot = np.logspace(np.log10(omega_range_global[0]), 
                            np.log10(omega_range_global[1]), 10000)
    
    if combined_loss:
        # For combined loss, plot single fitted noise spectrum
        optimized_args_combined = results_dict[n_values[0]]["optimized_args"]  # Same for all n
        noise_spectrum_fitted = noise_spectrum_combination(omega_plot, *optimized_args_combined)
        
        # Format parameters for legend
        param_str = format_parameters(optimized_args_combined)
        label_str = f"Combined Fit S(ω)"
        if param_str:
            label_str += f" ({param_str})"
        
        ax2.plot(omega_plot, noise_spectrum_fitted, 'black', linewidth=3, 
                label=label_str)
        
        # Plot original noise spectrum if available (synthetic data)
        if not experimental_data and noise_params is not None:
            noise_spectrum_original = noise_spectrum_combination(omega_plot, *noise_params)
            orig_param_str = format_parameters(noise_params)
            orig_label = f"Original S(ω)"
            if orig_param_str:
                orig_label += f" ({orig_param_str})"
            ax2.plot(omega_plot, noise_spectrum_original, 'red', linewidth=2, linestyle='--',
                    label=orig_label)
    
    # Plot noise inversion for each N
    for i, n in enumerate(n_values):
        # Get data from results
        t_points = results_dict[n]["t_points"]

        color = colors[i % len(colors)]
        
        if "C(t)_dirt" in results_dict[n]:
            C_t_observed = results_dict[n]["C(t)_dirt"]
        else:
            C_t_observed = results_dict[n]["C(t)_true"]
        
        # Calculate omega values for this n
        omega_n = n * np.pi / t_points
        
        if include_inversion:
            # Calculate noise inversion using delta approximation. This will show what C(t) would be if delta approximation was used.
            try:
                noise_inv = noise_inversion_delta(t_points, C_t_observed)
                # Filter out infinite and invalid values
                valid_mask = np.isfinite(noise_inv) & (noise_inv > 0)
                omega_n_valid = omega_n[valid_mask]
                noise_inv_valid = noise_inv[valid_mask]
                
                if len(noise_inv_valid) > 0:
                    ax2.plot(omega_n_valid, noise_inv_valid, 
                            linestyle='--', color=color, linewidth=2, alpha=0.8,
                            label=f'Noise Inversion N={n}')
            except Exception as e:
                print(f"Warning: Could not compute noise inversion for N={n}: {e}")
                continue
        
        if not combined_loss:

            # For individual fits, plot the fitted noise spectrum for each n
            optimized_args_n = results_dict[n]["optimized_args"]

            C_t_fitted = func_to_fit(t_points, noise_spectrum_combination, 
                                *optimized_args_n, delta=delta_approx, **results_dict[n]["finite_width_params"])
            
            sig = np.std(C_t_observed-C_t_fitted)
            print(f"σ for C(t) fit, N={n}: {sig:.6f}")

            t_points_truncated = t_points[find_widest_contiguous_stretch(C_t_observed, sig, 1-sig)]
            omega_truncated = n * np.pi / t_points_truncated

            # noise_spectrum_n = ns.noise_spectrum_combination(omega_plot, *optimized_args_n)
            # noise_spectrum_n = ns.noise_spectrum_combination(omega_n_valid, *optimized_args_n)
            noise_spectrum_n = noise_spectrum_combination(omega_truncated, *optimized_args_n)

            # Format parameters for legend
            param_str_n = format_parameters(optimized_args_n)
            label_str_n = f"Fitted S(ω) N={n}"
            if param_str_n:
                label_str_n += f" ({param_str_n})"
            
            # ax2.plot(omega_plot, noise_spectrum_n, color=color, linewidth=2,
            #         label=label_str_n)
            # ax2.plot(omega_n_valid, noise_spectrum_n, color=color, linewidth=2,
            #         label=label_str_n)
            ax2.plot(omega_truncated, noise_spectrum_n, color=color, linewidth=2,
                    label=label_str_n)

    # Add original noise spectrum for individual fits if available
    if not combined_loss and not experimental_data and noise_params is not None:
        noise_spectrum_original = noise_spectrum_combination(omega_plot, *noise_params)
        orig_param_str = format_parameters(noise_params)
        orig_label = f"Original S(ω)"
        if orig_param_str:
            orig_label += f" ({orig_param_str})"
        ax2.plot(omega_plot, noise_spectrum_original, 'red', linewidth=2, linestyle=':',
                label=orig_label)
    
    # Place legend outside plot on the right
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for the legend

    print()
    print("True Parameters Used:")
    print(f"  Noise Spectrum Parameters: {noise_params if noise_params else 'None'}")
    print(f"  Delta Approximation: {'Enabled' if delta_approx else 'Disabled'}")
    print(f"  Scaling Factor F: {F}")
    # Print optimized parameters
    if combined_loss:
        print("\nOptimized Parameters from Combined Fit:")
        optimized_args_combined = results_dict[n_values[0]]["optimized_args"]
        for i, arg_dict in enumerate(optimized_args_combined):
            print(f"Parameter set {i+1}:")
            for key, value in arg_dict.items():
                if isinstance(value, list):
                    print(f"  {key}: {[f'{v:.6f}' for v in value]}")
                else:
                    print(f"  {key}: {value:.6f}")
    else:
        print("\nOptimized Parameters for Individual Fits:")
        for n in n_values:
            print(f"\nN={n}:")
            optimized_args_n = results_dict[n]["optimized_args"]
            for i, arg_dict in enumerate(optimized_args_n):
                print(f"  Parameter set {i+1}:")
                for key, value in arg_dict.items():
                    if isinstance(value, list):
                        print(f"    {key}: {[f'{v:.6f}' for v in value]}")
                    else:
                        print(f"    {key}: {value:.6f}")
    
    # Calculate and print total MSE
    total_mse = 0
    individual_mses = {}
    for n in n_values:
        t_points = results_dict[n]["t_points"]
        finite_width_params = results_dict[n]["finite_width_params"]
        optimized_args = results_dict[n]["optimized_args"]
        
        if "C(t)_dirt" in results_dict[n]:
            C_t_observed = results_dict[n]["C(t)_dirt"]
        else:
            C_t_observed = results_dict[n]["C(t)_true"]
        
        C_t_pred = func_to_fit(t_points, noise_spectrum_combination, 
                              *optimized_args, delta=delta_approx, **finite_width_params)
        
        if not experimental_data:
            C_t_pred = F * C_t_pred
            
        individual_mses[n] = mean_squared_error(C_t_observed, C_t_pred)
        total_mse += individual_mses[n]
    
    print(f"\nIndividual MSEs: {individual_mses}")
    print(f"Total MSE: {total_mse:.6f}")
    
    # Save plot if directory provided
    if save_dir is not None:
        plot_filename = "combined_analysis"
        if combined_loss:
            plot_filename += "_combined_loss"
        if delta_approx:
            plot_filename += "_delta_approx"
        plot_filename += ".png"
        
        plt.savefig(os.path.join(save_dir, plot_filename), dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {os.path.join(save_dir, plot_filename)}")
    
    plt.show()
    # return fig

def get_bootstrap_sample(data: Dict, used_combinations: Set[Tuple] = None) -> Dict[int, List[float]]:
    """
    Randomly select a unique combination of runs for each N value for bootstrapping.
    
    Args:
        data: The main data dictionary
        used_combinations: Set of previously used combinations (tuples of run keys)
    
    Returns:
        Dictionary in format {n_value: [A_value, alpha_value]}
    """
    if used_combinations is None:
        used_combinations = set()
    
    # Get available runs for each N value
    available_runs = {}
    for n_value in data.keys():
        available_runs[n_value] = list(data[n_value].keys())
    
    # Try to find a unique combination
    max_attempts = 10000  # Prevent infinite loops
    attempts = 0
    
    while attempts < max_attempts:
        # Randomly select one run for each N value
        selected_runs = {}
        combination_tuple = []
        
        for n_value in sorted(data.keys()):
            selected_run = random.choice(available_runs[n_value])
            selected_runs[n_value] = selected_run
            combination_tuple.append(selected_run)
        
        combination_tuple = tuple(combination_tuple)
        
        # Check if this combination has been used before
        if combination_tuple not in used_combinations:
            used_combinations.add(combination_tuple)
            
            # Build the result dictionary
            result = {}
            for n_value in sorted(data.keys()):
                run_key = selected_runs[n_value]
                run_data = data[n_value][run_key]
                result[n_value] = [run_data["A"], run_data["alpha"]]
            
            return result
        
        attempts += 1
    
    raise ValueError(f"Could not find unique combination after {max_attempts} attempts. "
                     f"You may have exhausted all possible combinations.")

def bootstrap_multiple_samples(data: Dict, num_bootstrap: int) -> List[Dict[int, List[float]]]:
    """
    Generate multiple unique bootstrap samples.
    
    Args:
        data: The main data dictionary
        num_bootstrap: Number of bootstrap samples to generate
    
    Returns:
        List of bootstrap sample dictionaries
    """
    used_combinations = set()
    bootstrap_samples = []
    
    for i in range(num_bootstrap):
        try:
            sample = get_bootstrap_sample(data, used_combinations)
            bootstrap_samples.append(sample)
            # print(f"Bootstrap sample {i+1}/{num_bootstrap} generated")
        except ValueError as e:
            print(f"Warning: {e}")
            print(f"Generated {len(bootstrap_samples)} unique samples out of {num_bootstrap} requested")
            break
    
    return bootstrap_samples

# Calculate total possible combinations
def calculate_total_combinations(data: Dict) -> int:
    """Calculate the total number of possible unique combinations."""
    total = 1
    for n_value in data.keys():
        total *= len(data[n_value])
    return total