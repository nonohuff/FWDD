#!/bin/bash
#SBATCH --job-name=train_small_model_n_128_tau_p_0.024_no_maxpool
#SBATCH --time=6:00:00
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --mem=100GB
#SBATCH --mail-user=nhuffman@stanford.edu
#SBATCH --mail-type=ALL

module load viz

## module load py-matplotlib/3.4.2_py39
## module load py-tensorflow/2.10.0_py39
## module load py-scipy/1.10.1_py39
## module load py-numpy/1.24.2_py39
## module load cuda/12.6.1
## module load cudnn/9.4.0
## module load cuda/11.2.0
## module load cudnn/8.1.1.33
## module load py-scikit-learn/1.3.2_py39

ml python/3.12.1
ml cudnn/8.9.0.131
ml cuda/12.2.0
python3 -m venv /scratch/users/nhuffman/noise_spectrum_mitigation/venv/tf_env
source /scratch/users/nhuffman/noise_spectrum_mitigation/venv/tf_env/bin/activate
pip install tensorflow[and-cuda]==2.17.0
pip install numpy matplotlib scipy scikit-learn joblib


python3 train_small_model_n_128_tau_p_0.024_no_maxpool.py