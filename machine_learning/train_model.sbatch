#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --time=1-0:00:00
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -G 1
#SBATCH -C "GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --nodes 1
#SBATCH --mem=20GB
#SBATCH --mail-user=nhuffman@stanford.edu
#SBATCH --mail-type=ALL

if [ $# -ne 3 ]; then
    echo "Error: Expected 3 arguments"
    echo "Usage: sbatch $0 N tau_p Gaussian_noise"
    echo "N: Number of Pi pulses in the CPMG sequence (int)"
    echo "tau_p: Width of Pi pulses in the CPMG sequence (float)"
    echo "Gaussian_noise: Boolean indicating whether to use Gaussian noise (bool)"
    exit 1

fi

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
# Only need to un-comment the following line when initializing the virtual environment. If you have done that already, you can comment it out again.
# python3 -m venv /scratch/users/nhuffman/noise_spectrum_mitigation/venv/tf_env
source /scratch/users/nhuffman/noise_spectrum_mitigation/venv/tf_env/bin/activate
pip install tensorflow[and-cuda]==2.17.0
pip install numpy matplotlib scipy scikit-learn joblib numba


python3 train_model.py "$1" "$2" "$3"