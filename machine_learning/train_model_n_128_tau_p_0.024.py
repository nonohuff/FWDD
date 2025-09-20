import os
import psutil
import random
from math import ceil, fsum
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import time

from scipy.signal import find_peaks
from scipy.integrate import quad, simpson, trapezoid, IntegrationWarning

from scipy.stats import cauchy

import joblib
from joblib import Parallel, delayed, parallel_backend

from tqdm import tqdm, trange
from tqdm.contrib import tenumerate

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from itertools import product

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
    
    return 8 * (np.sin((omega*T) / 2) ** 2) * (np.sin((omega*T)/(4*N))**4) / (np.cos((omega*T)/(2*N))**2)

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
        t_k = np.linspace((T/N-tau_p)/2, T*(1-1/(2*N))-(tau_p/2), N)  # Pulses are evenly spaced
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

# from tensorflow.python.client import device_lib

# device_lib.list_local_devices()

def build_cnn(input_shape, output_shape, activation_output='linear', kernel_size=10):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Encoder: 3 Conv1D + BatchNorm + MaxPooling + Dropout
    x = layers.Conv1D(filters=1000, kernel_size=kernel_size, padding='same')(inputs)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)  # ReLU activation after BatchNorm
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=2000, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=3000, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=4000, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=5000, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Decoder: 3 Conv1D + BatchNorm + UpSampling + Dropout
    x = layers.Conv1D(filters=5000, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.UpSampling1D(size=2)(x)
    
    # Decoder: 3 Conv1D + BatchNorm + UpSampling + Dropout
    x = layers.Conv1D(filters=4000, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.UpSampling1D(size=2)(x)
    
    x = layers.Conv1D(filters=3000, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.UpSampling1D(size=2)(x)
    
    x = layers.Conv1D(filters=2000, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.UpSampling1D(size=2)(x)

    x = layers.Conv1D(filters=1000, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.UpSampling1D(size=2)(x)
    
    # Dense layer to match output shape
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_shape[-1], activation=activation_output)(x)
    
    # Model definition
    model = models.Model(inputs, outputs)
    
    return model

def add_gaussian_noise(C_t, loc, sigma, copies):
    """
    Create multiple copies of each row in C_t and add different Gaussian noise to each copy.
    
    Parameters:
    -----------
    C_t : numpy.ndarray
        Input array of shape (n,k) where n is the number of rows 
        and k is the number of columns
    loc : float
        Mean of the Gaussian noise
    sigma : float
        Standard deviation of the Gaussian noise
    copies : int
        Number of noisy copies to create for each row
        
    Returns:
    --------
    numpy.ndarray
        Array of shape (n*copies, k) where each original row is repeated
        copies times with different noise added
    """
    n, k = C_t.shape
    
    # Repeat each row copies times
    repeated = np.repeat(C_t, copies, axis=0)
    
    # Generate noise for all copies
    noise = np.random.normal(loc, sigma, size=(n*copies, k))
    
    return repeated + noise



##### Define Parameters #####

# number of points for the resolution of S(w). This will be the output dimention of the NN.
frequency_resolution = 1000

gaussian_noise = True
copies = 6 # number of times to add gaussian noise to the data
mu = 0
sigma = 0.07

np_data_type = np.float64
finite_width_params = {
    "N": 128, #CPMG-N (test spin echo)
    "tau_p": 24*10**(-3), #pi pulse width in mircoseconds
    "integration_method": "trapezoid", # The method to use for integration. Options are "quad","trapezoid",and "simpson".
    "omega_resolution": int(10**5), # 6*10**5 The number of points to use when integrating to yield the coherence decay profile, C(t), unless method is "quad", in which case, this is the resolution used to determine the filter function and integrand outputs.
    "omega_range" : (10**(-4),10**8), # The number of peaks to use in the filter function
    #"omega_range" : (10**(-4),10**5)
    "num_peaks_cutoff": 100, # must be >=0 and <num_peaks. If this value is set, then only the top num_peaks_cutoff peaks will be used in the integration. This can be useful for speeding up the integration when the noise spectrum has many peaks.
    "peak_resolution": 100, # For every method except "quad", you can specify additional points around the singular points of the filter function. This argument specifies the number of points to use around each singular point.
    # "quad_limit": 10**5, # The maximum number of subdivisions in the adaptive quadrature algorithm
    # "epsabs": 1.49e-8, # Absolute tolerance for the quadrature integration
    # "epsrel": 1.49e-8, # Relative tolerance for the quadrature integration
}

t_points = np.logspace(np.log10(finite_width_params["N"]*finite_width_params["tau_p"]+1e-10), np.log10(300), 201, dtype=np_data_type)  # Time points for coherence profile. Units are Microseconds

assume_delta = False # Whether or not to approximate Filter function pulses as Delta functions

if assume_delta == True:
    X_1f, Y_1f = joblib.load(global_filepath + 'training_data_' + noise_spectrum_1f.__name__ + '_delta.pkl')["data"]
    X_lor, Y_lor = joblib.load(global_filepath + 'training_data_' + noise_spectrum_lor.__name__ + '_delta.pkl')["data"]
    # X_combo, Y_combo = joblib.load(global_filepath + 'training_data_' + noise_spectrum_combination.__name__ + '_delta.pkl')["data"]
else:
    params = '_'.join(f"{k}_{v}" for k, v in finite_width_params.items())

    noise_types = [
        "noise_spectrum_1f",
        "noise_spectrum_lor", 
        "noise_spectrum_combo_N1f_1_Nlor_1_NC_1",
        "noise_spectrum_combo_N1f_1_Nlor_2_NC_1"
    ]

# Initialize X and Y as None
X = None
Y = None

# Handle all datasets in the loop
om = np.logspace(np.log10(finite_width_params["omega_range"][0]),np.log10(finite_width_params["omega_range"][1]),frequency_resolution)
for name in noise_types:
    data_X = joblib.load(f"{global_filepath}training_data_{name}_{params}_finite_width.pkl")["data"]
    noise_profile_args = joblib.load(f"{global_filepath}training_data_{name}_{params}_finite_width.pkl")["noise_profile_args"]
    data_Y = np.empty((data_X.shape[0],len(om)))
    if name == "noise_spectrum_1f":
        for i in range(data_X.shape[0]):
            data_Y[i,:] = noise_spectrum_1f(om,*noise_profile_args[i])
    elif name == "noise_spectrum_lor": 
        for i in range(data_X.shape[0]):
            data_Y[i,:] = noise_spectrum_lor(om,*noise_profile_args[i])
    else:
        for i in range(data_X.shape[0]):
            data_Y[i,:] = noise_spectrum_combination(om,*noise_profile_args[i])

    if X is None:
        X = data_X
        Y = data_Y
    else:
        X = np.concatenate((X, data_X), axis=0)
        Y = np.concatenate((Y, data_Y), axis=0)

del data_X
del data_Y # erase varible to free up memory

# Add Gaussian noise
if gaussian_noise:
    X = add_gaussian_noise(X, mu, sigma,copies)
    Y = np.repeat(Y, copies, axis=0)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

model_size = Y.shape[1]
XTraining, XValidation, YTraining, YValidation = train_test_split(X,np.log1p(Y),shuffle=True,test_size=0.1) # before model building

print(f"XTraining shape: {XTraining.shape}")
print(f"YTraining shape: {YTraining.shape}")
print(f"XValidation shape: {XValidation.shape}")
print(f"YValidation shape: {YValidation.shape}")
# Free up memory
del X
del Y

# # Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=1000,
    decay_rate=0.9
)

# Model Parallelism
# First configure memory growth for GPUs
gpus = tf.config.list_physical_devices('GPU')
# devices = [f'/gpu:{i}' for i in range(len(gpus))]
if not gpus:
    print("No GPUs found!")
else:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
        gpu_info = tf.config.experimental.get_device_details(gpu)
        gpu_name = gpu_info.get('device_name', 'Unknown GPU')
        print(f"GPU: {gpu_name}")
        try:
            gpu_memory = tf.config.experimental.get_memory_usage(gpu)
            print(f"Memory: {gpu_memory}")
        except Exception as e:
            print("Issue with memory retrieval:",e)
        

# Summary of GPUs detected
print(f"\nTotal GPUs detected: {len(gpus)}")


model = build_cnn((len(t_points), 1), (model_size,),activation_output='linear',kernel_size=3)
model.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=lr_schedule))
model.summary()


# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# warm-up the model
for _ in range(3):
    model.predict(XTraining)  # or model.train_on_batch(XTraining, YTraining)
# run the model
history = model.fit(XTraining,YTraining,batch_size=64,epochs=500,validation_data=(XValidation,YValidation),callbacks=[early_stopping])
##### 9/4/2024 NOTE: The above uses logs bc Souvik encountered issues with e**(-chi(t)) and S(omega) both being exponential and having both very large and very small values.
# He read that log'ing them would help. Unclear if this works.#####



# # Plot training history
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.yscale('log')


if gaussian_noise:
    model.save(global_filepath + 'trained_model_n_'+str(finite_width_params["N"])+"_tau_p_"+str(finite_width_params["tau_p"])+'_noise_mu_'+str(mu)+'_sigma_'+str(sigma)+'.keras')
    print("Model saved to " + global_filepath + 'trained_model_n_'+str(finite_width_params["N"])+"_tau_p_"+str(finite_width_params["tau_p"])+'_noise_mu_'+str(mu)+'_sigma_'+str(sigma)+'.keras')
    plt.savefig(global_filepath + 'training_history_model_n_'+str(finite_width_params["N"])+"_tau_p_"+str(finite_width_params["tau_p"])+'_noise_mu_'+str(mu)+'_sigma_'+str(sigma)+'.png')  # Save the figure
    print("Training history figure saved to " + global_filepath + 'training_history_model_n_'+str(finite_width_params["N"])+"_tau_p_"+str(finite_width_params["tau_p"])+'_noise_mu_'+str(mu)+'_sigma_'+str(sigma)+'.png')
else:
    model.save(global_filepath + 'trained_model_n_'+str(finite_width_params["N"])+"_tau_p_"+str(finite_width_params["tau_p"])+'.keras')
    print("Model saved to "+global_filepath + 'trained_model_n_'+str(finite_width_params["N"])+"_tau_p_"+str(finite_width_params["tau_p"])+'.keras')
    plt.savefig(global_filepath + 'training_history_model_n'+str(finite_width_params["N"])+"_tau_p_"+str(finite_width_params["tau_p"])+'.png')  # Save the figure
    # plt.show()
    print("Training history figure saved to " + global_filepath + 'training_history_model_n'+str(finite_width_params["N"])+"_tau_p_"+str(finite_width_params["tau_p"])+'.png')