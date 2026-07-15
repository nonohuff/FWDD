import numpy as np
import numba as nb

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
        # t_k = np.linspace((T/N-tau_p)/2, T*(1-1/(2*N))-(tau_p/2), N)  # Pulses are evenly spaced. This is the time of beginning of each pulse. This should not be used, but I'm leaving in as a comment in case you need the beggining pulse timings for something else. 
        t_k = T/(2*N)*np.arange(1,2*N+1,2) # Pulses are evenly spaced. This is the middle of the each pulse. Alternatively, np.linspace((T/N)/2, T*(1-1/(2*N)), N)
        if (t_k<0).any():
            raise ValueError("One of the Pulse start times is negative. This is not allowed.")
        
    if method == 'numba':
        # Uses numba to speed up the sum. Should be the fastest implementation.
        # Doesn't store the intermediate results, so it's memory efficient.
        sum_term = numba_complex_sum(t_k, omega)
    elif method == 'numpy_array':
        # Uses numpy array broadcasting to vectorize the sum. 
        # A little faster than a summation loop, and more numerically stable, but has a higher memory overhead. 
        sum_array = np.exp(1j * omega[:, np.newaxis] * t_k)
        sum_array[:,::2] *= -1 # Adds negative sign to odd indices of t_k
        # sum_array.sort(axis=1)
        sum_term = np.sum(sum_array, axis=1)
    elif method == "numpy":
        # Uses a for loop to sum the terms. Slowest implementation, but more memory efficient than the array version.
        # You can instead sum the positive and negative terms separately, then subtract them, for a slightly more numerically stable result (see below).

        # neg_sum_term = np.zeros(omega.shape)
        # pos_sum_term = np.zeros(omega.shape)
        # for k in range(0,N,2):
        #     # neg_sum_term = neg_sum_term - np.exp(1j * omega * t_k[k])
        #     neg_sum_term = np.sum(np.vstack((neg_sum_term,np.exp(1j * omega * t_k[k]))),axis=0)
            
        # for k in range(1,N+1,2):
        #     # pos_sum_term = pos_sum_term + np.exp(1j * omega * t_k[k])
        #     pos_sum_term = np.sum(np.vstack((pos_sum_term,np.exp(1j * omega * t_k[k]))),axis=0)

        # sum_term = (pos_sum_term-neg_sum_term)
            
        sum_term = np.zeros(omega.shape, dtype=np.complex128)
        for k in range(N):
            sum_term += ((-1)**(k+1)) * np.exp(1j * omega * t_k[k])
    
    result = np.power(np.abs(1 + np.power(-1,N+1) * np.exp(1j * omega * T) + 2 * np.cos(omega * tau_p / 2) * sum_term),2)
    
    return result