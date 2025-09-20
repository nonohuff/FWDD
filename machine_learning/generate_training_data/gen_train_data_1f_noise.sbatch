#!/bin/bash
#SBATCH --job-name=gen_train_data_1f_noise
#SBATCH --time=1-0:00:00
#SBATCH -p bigmem
#SBATCH -c 64
#SBATCH --mem=256GB
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

python3 gen_train_data_1f_noise.py "$1" "$2"
