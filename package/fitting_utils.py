import numpy as np
from scipy.optimize import brentq
from .noise_spectra import noise_spectrum_combination
from .noise_learning_fitting import func_to_fit


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