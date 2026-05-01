#!/bin/bash

#SBATCH --array=0-127
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=7G
#SBATCH -J save
#SBATCH -t 03:00:00
#SBATCH -o logs/L200m6/aniso-results-%J.out

flight start
flight env activate gridware

module purge
module load apps/python3/3.12.7/gcc-8.5.0
module list

source ~/jupyter_project/swift-env/bin/activate

python3 compute_anisotropies.py