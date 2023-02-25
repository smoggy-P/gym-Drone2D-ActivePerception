#!/bin/sh
#
#SBATCH --job-name="run"
#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

module load 2022r2
module load python/3.8.12
module load py-matplotlib
module load py-numpy
module load openmpi py-torch
module load py-scikit-learn
srun python main.py 
