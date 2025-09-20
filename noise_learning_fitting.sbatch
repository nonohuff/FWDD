#!/bin/bash
#SBATCH --job-name=gradient_descent_tests
#SBATCH --output=slurm_out_%A_%a.out      # Standard output file
#SBATCH --partition=normal
#SBATCH --array=0                       # Array range for 1 task(s). 
                                        # If you want to do Bootstrapping, you will need to change noise_learning_fitting.py (preferrably make a new file) such that experimental_data = False, noise =True, and noise_level = sigma. 
                                        # Then, instead of generating random parameters for C(t), use the parameters you found from fitting your experimental data.
                                        # Finally, all you need to do is make change the --array value to the number of bootstrap samples you want. For example, if you want 100 bootstrap samples, change --array=0 to --array=0-99.
#SBATCH --ntasks=1                        # Number of tasks per job
#SBATCH --mem=64G                         # Memory per task
#SBATCH --cpus-per-task=30                # Number of CPU cores per task
#SBATCH --time=5:00:00                    # Time limit per task
#SBATCH --mail-user=nhuffman@stanford.edu
#SBATCH --mail-type=ALL


module load viz

module load py-scipy/1.12.0_py312
module load py-numpy/1.26.3_py312
module load py-scikit-learn/1.5.1_py312
module load py-numba/0.60.0_py312
module load py-matplotlib/3.8.3_py312
module load py-pandas/2.2.1_py312

module load python/3.12.1

python3 -m pip install scikit-optimize

# Agruments for the script
# $1: tau_pi value in (microseconds) - float
# $2: N1f, # Number of 1/f noise parameters - int
# $3: Nlor, # Number of Lorentzian noise parameters - int
# $4: NC, # Number of white noise parameters - int
# $5: Ncom_frac, # Number of combined fraction noise parameters - int
# $6: delta_approx - Boolean (True/False)
# $7: noise present? - Boolean (True/False)
# $8: noise_level - float
# $9: Max number of optimization iterations - int (the more parameters your noise model has, the more iterations you will need)
# $10: Max population size for optimization - int (the wider the search space, the larger the population size you will need)
# $11: Global Scaling, F - float (defult is 1)
# $12: Use Souvik's experimental data? - Boolean (True/False)
# $13: Use combined loss across all n values? - Boolean (True/False)
# $14: C_t_min - float (default is -2.0)
# $15: C_t_max - float (default is 2.0)


python3 noise_learning_fitting.py "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}"