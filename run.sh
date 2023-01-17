#!/bin/sh
#
#SBATCH --job-name="run"
#SBATCH --partition=memory
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-3me-msc-ro

module load 2022r2
module load python/3.8.12
module load py-numpy
module load openmpi py-torch
python main.py