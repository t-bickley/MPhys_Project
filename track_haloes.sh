#!/bin/bash

#SBATCH --array=0-127
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH -J save
#SBATCH -t 01:30:00
#SBATCH -o logs/L025m5_updated/example-halo-aniso-results-%J_%a.out

flight start
flight env activate gridware

module purge
module load apps/python3/3.12.7/gcc-8.5.0
module list

source ~/jupyter_project/swift-env/bin/activate

python3 L025m5_evo.py