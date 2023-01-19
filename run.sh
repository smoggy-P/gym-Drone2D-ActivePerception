#!/bin/sh
#
#SBATCH --job-name="run"
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH --time=30 # default in minutes
#SBATCH --account=education-3me-msc-ro

module load 2022r2
module load python/3.8.12
module load py-matplotlib
module load py-numpy
module load openmpi py-torch
srun python train.py