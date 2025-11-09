#!/bin/bash
#SBATCH --job-name test
#SBATCH --nodes 1
#SBATCH --output test.out
#SBATCH --error test.err
#SBATCH -p high
#SBATCH --cpus-per-task 2
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user butlerj@berkeley.edu
export OMP_NUM_THREADS=1
cd ..
conda run -n antarctic_ars python cross_validation_xgb.py --x_cols max_ocean_SLP_gradient max_landfalling_v850hPa max_landfalling_omega850 max_IWV_ais cumulative_landfalling_area max_south_extent --y_col cumulative_snowfall_ais --hyperparam_json test_hyperparams.json --chunk_size 8  --save_name test.csv