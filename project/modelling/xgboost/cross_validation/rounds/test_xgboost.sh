#!/bin/bash
#SBATCH -A m3318
#SBATCH --job-name=coverage_simulations
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --output=coverage_simulations%j.out
#SBATCH --error=coverage_simulations%j.err
#SBATCH --exclusive
#SBATCH -q regular
#SBATCH --ntasks-per-node=64
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=butlerj@berkeley.edu
#SBATCH --constraint=cpu