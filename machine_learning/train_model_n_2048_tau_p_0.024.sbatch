#!/bin/bash
#SBATCH --job-name=train_model_n_128_tau_p_0.024
#SBATCH --time=1-12:00:00
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 3
#SBATCH -C GPU_MEM:48GB
#SBATCH --mem=100GB
#SBATCH --mail-user=nhuffman@stanford.edu
#SBATCH --mail-type=ALL

module load viz

module load py-matplotlib/3.4.2_py39
module load py-tensorflow/2.10.0_py39
# module load py-scipy/1.10.1_py39
# module load py-numpy/1.24.2_py39
# module load cuda/12.6.1
# module load cudnn/9.4.0
module load cuda/11.2.0
module load cudnn/8.1.1.33
# module load py-scikit-learn/1.3.2_py39


python3 train_model_n_2048_tau_p_0.024.py