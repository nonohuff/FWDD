#!/bin/bash
#SBATCH --job-name=merge_and_cleanup_data_files      # Job name
#SBATCH --time=10:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=25GB
#SBATCH --mail-user=nhuffman@stanford.edu
#SBATCH --mail-type=ALL

module load py-numpy/1.24.2_py39

python3 merge_and_cleanup_data_files.py