#!/bin/bash
#SBATCH --job-name=lor_training_data_job      # Job name
#SBATCH --output=slurm_out_%A_%a.out      # Standard output file
#SBATCH --partition=normal
#SBATCH --array=0-999%100                      # Array range for 1000 tasks, no more than 100 simultaneous jobs
#SBATCH --ntasks=1                       # Number of tasks per job
#SBATCH --mem=20G                         # Memory per task
#SBATCH --cpus-per-task=10                # Number of CPU cores per task
#SBATCH --time=2:30:00                 # Time limit per task
#SBATCH --mail-user=nhuffman@stanford.edu
#SBATCH --mail-type=ALL

if [ $# -ne 2 ]; then
    echo "Error: Expected 2 arguments"
    echo "Usage: sbatch $0 N tau_p"
    exit 1

fi

module load viz

module load py-scipy/1.10.1_py39
module load py-numpy/1.24.2_py39
module load py-scikit-learn/1.3.2_py39
module load py-numba/0.59.1_py39
module load py-matplotlib/3.7.1_py39
module load py-pandas/2.0.1_py39
# module load py-h5py/3.7.0_py39

module load python/3.9.0

python3 -m pip install tables

# # First input on the command line is an integer N, which corresponds to the number of Pi pulses in the cpmg sequence
# # Second input on the command line is a float , tau_p, which corresponds to the width of Pi pulses in the cpmg sequence (in mircoseconds)
srun python3 gen_train_data_lor_noise_SLURMarray.py "$1" "$2"