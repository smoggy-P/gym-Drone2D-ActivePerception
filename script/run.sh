#!/bin/sh
#
#SBATCH --job-name="test_experiment"
#SBATCH --partition=compute
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=research-3me-cor

srun ./executable.x  