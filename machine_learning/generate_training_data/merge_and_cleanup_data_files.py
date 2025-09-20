import os
import sys
import glob
import joblib
import numpy as np

def combine_task_files(global_filepath, base_filename, task_filename_pattern):
    """
    Combines task-specific save files into the global save file and deletes task-specific files.

    Parameters:
    -----------
    global_filepath : str
        Directory where the global and task-specific files are stored.
    base_filename : str
        Name of the global save file.
    task_filename_pattern : str
        Pattern to match task-specific files (e.g., '*_task_*.pkl').

    Returns:
    --------
    None
    """
    global_save_path = os.path.join(global_filepath, base_filename)
    task_files = glob.glob(os.path.join(global_filepath, task_filename_pattern))

    print(f"Found {len(task_files)} task files to merge.")

    # Load existing global data if it exists
    try:
        global_data = joblib.load(global_save_path)
        global_X = list(global_data["data"][0])
        global_Y = list(global_data["data"][1])
        global_params = list(global_data["noise_profile_args"])
        print(f"Loaded existing global data with {len(global_params)} parameter sets.")
    except FileNotFoundError:
        global_data = {}
        global_X, global_Y, global_params = [], [], []
        print("No existing global data found. Creating a new file.")

    # Merge task files
    try:
        for task_file in task_files:
            print(f"Merging {task_file}...")
            task_data = joblib.load(task_file)
            global_X.extend(task_data["data"][0])
            global_Y.extend(task_data["data"][1])
            global_params.extend(task_data["noise_profile_args"])
    
        # Save the merged data back to the global file
        merged_data = {
            "noise_profile": global_data.get("noise_profile") if global_X else task_data["noise_profile"],
            "noise_profile_args": np.array(global_params),
            "filter_function_args": global_data.get("filter_function_args", task_data["filter_function_args"]),
            "data": (np.array(global_X), np.array(global_Y)),
        }
        print(f"Saving merged data to {global_save_path}...")
        print(f"C(t) training examples shape: {merged_data['data'][0].shape}")
        joblib.dump(merged_data, global_save_path)

        # Delete task files
        for task_file in task_files:
            print(f"Deleting {task_file}...")
            os.remove(task_file)

        print("All task files merged and cleaned up successfully.")
    except Exception as e:
        print(f"Error merging task files: {e}")
        return


# Example Usage
# if __name__ == "__main__":

# base_filename = "training_data_noise_spectrum_1f_finite_width.pkl"  # Global save file name
# task_filename_pattern = "training_data_noise_spectrum_1f_task_*.pkl"  # Task file pattern

# global_filepath = str(sys.argv[1])
# base_filename = str(sys.argv[2])
global_filepath = "./training_data/"  # Path to directory with data files
# base_filename = "training_data_noise_spectrum_1f_N_512_tau_p_0.024_integration_method_trapezoid_omega_resolution_100000_omega_range_(0.0001, 100000000)_num_peaks_cutoff_100_peak_resolution_100_finite_width.pkl"
# base_filename = "training_data_noise_spectrum_lor_N_128_tau_p_0.024_integration_method_trapezoid_omega_resolution_100000_omega_range_(0.0001, 100000000)_num_peaks_cutoff_100_peak_resolution_100_finite_width.pkl"
base_filename = "training_data_noise_spectrum_combo_N1f_1_Nlor_2_NC_1_N_128_tau_p_0.024_integration_method_trapezoid_omega_resolution_100000_omega_range_(0.0001, 100000000)_num_peaks_cutoff_100_peak_resolution_100_finite_width.pkl"
task_filename_pattern = base_filename.replace('_finite_width.pkl', "_task_*.pkl")

combine_task_files(global_filepath, base_filename, task_filename_pattern)
