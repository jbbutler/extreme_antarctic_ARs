#!/bin/bash
#SBATCH --job-name snow_xgb_round1
#SBATCH --nodes 1
#SBATCH --output snow_xgb_round1.out
#SBATCH --error snow_xgb_round1.err
#SBATCH -p epurdom
#SBATCH --cpus-per-task 10
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user butlerj@berkeley.edu
export OMP_NUM_THREADS=1
cd ..
conda run -n antarctic_ars python cross_validation_xgb.py --x_cols max_ocean_SLP_gradient max_landfalling_v850hPa max_landfalling_omega850 max_IWV_ais cumulative_landfalling_area max_south_extent --y_col cumulative_snowfall_ais --hyperparam_json snow_round1_hyperparams.json --chunk_size 100  --save_name snow_xgb_round1.csv