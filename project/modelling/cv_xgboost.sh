#!/bin/bash
#SBATCH --job-name=xgb_snow_cv
#SBATCH --mail-type=ALL
#SBATCH --mail-user=butlerj@berkeley.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH -o cv.out #File to which standard out will be written
#SBATCH -e cv.err #File to which standard err will be written

source ~/.bashrc
conda activate antarctic_ars

python cross_validation.py
