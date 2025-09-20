import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
import time

import joblib
from joblib import Parallel, delayed, parallel_backend

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

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

# from tensorflow.python.client import device_lib

# device_lib.list_local_devices()

def build_cnn(input_shape, output_shape, activation_output='linear', kernel_size=3):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Encoder: 3 Conv1D + BatchNorm + MaxPooling + Dropout
    x = layers.Conv1D(filters=400, kernel_size=kernel_size, padding='same')(inputs)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)  # ReLU activation after BatchNorm
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=600, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(filters=800, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Decoder: 3 Conv1D + BatchNorm + UpSampling + Dropout
    x = layers.Conv1D(filters=800, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.UpSampling1D(size=2)(x)
    
    x = layers.Conv1D(filters=600, kernel_size=kernel_size, padding='same')(x)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.BatchNormalization()(x)  # BatchNorm after Conv
    # x = layers.ReLU()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.UpSampling1D(size=2)(x)
    
    x = layers.Conv1D(filters=400, kernel_size=kernel_size, padding='same')(x)
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

np_data_type = np.float64
finite_width_params = {
    "N": 2048, #CPMG-N (test spin echo)
    "tau_p": 24*10**(-3), #pi pulse width in mircoseconds
    "integration_method": "trapezoid", # The method to use for integration. Options are "quad","trapezoid",and "simpson".
    "omega_resolution": int(10**5), # 6*10**5 The number of points to use when integrating to yield the coherence decay profile, C(t), unless method is "quad", in which case, this is the resolution used to determine the filter function and integrand outputs.
    "omega_range" : (10**(-4),10**8), # The number of peaks to use in the filter function
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

    # Initialize with first dataset
    data = joblib.load(f"{global_filepath}training_data_{noise_types[0]}_{params}_finite_width.pkl")["data"]
    X, Y = data

# Append remaining datasets
for name in noise_types[1:]:
    data = joblib.load(f"{global_filepath}training_data_{name}_{params}_finite_width.pkl")["data"]
    X = np.concatenate((X, data[0]), axis=0)
    Y = np.concatenate((Y, data[1]), axis=0)

del data # erase varible to free up memory

model_size = Y.shape[1]
XTraining, XValidation, YTraining, YValidation = train_test_split(X,np.log1p(Y),shuffle=True,test_size=0.1) # before model building

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
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# devices = [f'/gpu:{i}' for i in range(len(gpus))]
if not gpus:
    print("No GPUs found!")
else:
    for gpu in gpus:
        gpu_info = tf.config.experimental.get_device_details(gpu)
        gpu_name = gpu_info.get('device_name', 'Unknown GPU')
        print(f"GPU: {gpu_name}")
        print(gpu_info)
# Summary of GPUs detected
print(f"\nTotal GPUs detected: {len(gpus)}")


# Create a MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Define options for automatic model parallelism
# options = tf.distribute.AutoStrategy(
#     num_replicas_in_sync=len(gpus),
#     cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
# )

# Create the model inside the strategy scope
with strategy.scope():
    model = build_cnn((len(t_points), 1), (model_size,), activation_output='linear', kernel_size=3)
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=lr_schedule)
    )


# Data Parallelizm 
# # Create a MirroredStrategy
# strategy = tf.distribute.MirroredStrategy()
# print(f'Number of devices: {strategy.num_replicas_in_sync}')

# # Create the model inside the strategy scope
# with strategy.scope():
#     model = build_cnn((len(t_points), 1), (model_size,), activation_output='linear', kernel_size=3)
#     model.compile(
#         loss='mean_squared_error',
#         optimizer=Adam(learning_rate=lr_schedule)
#     )


model.summary()


# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# warm-up the model
# for _ in range(3):
#     model.predict(X)  # or model.train_on_batch(X, Y)
# run the model
# Train the model (this will automatically use all available GPUs)
history = model.fit(
    XTraining, 
    YTraining,
    batch_size=64 * strategy.num_replicas_in_sync,  # Adjust batch size for multiple GPUs
    epochs=500,
    validation_data=(XValidation, YValidation),
    callbacks=[early_stopping]
)
##### 9/4/2024 NOTE: The above uses logs bc Souvik encountered issues with e**(-chi(t)) and S(omega) both being exponential and having both very large and very small values.
# He read that log'ing them would help. Unclear if this works.#####

# Save the trained model
model.save(global_filepath + 'trained_model.keras')

# # Plot training history
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.yscale('log')
plt.savefig(global_filepath + 'training_history.png')  # Save the figure
# plt.show()