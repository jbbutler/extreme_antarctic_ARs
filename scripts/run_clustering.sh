#!/bin/bash
#SBATCH --job-name=AR_catalog_run
#SBATCH --mail-type=ALL   

#SBATCH --mail-user=butlerj@berkeley.edu
#SBATCH -o catalog_run.out #File to which standard out will be written
#SBATCH -e catalog_run.err #File to which standard err will be written
python clustering.py --save_path '/scratch/users/butlerj/extreme_antarctic_ars/catalog_runs' --frac_perturbs 0.25 0.75 1 --time_perturbs 15 18 21 24 --minpts_perturbs 3 10 --reppts_perturbs 5 15 --seed_perturbs 1111 2222 3333 4444 5555 