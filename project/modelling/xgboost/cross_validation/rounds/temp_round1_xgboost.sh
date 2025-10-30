#!/bin/bash
#SBATCH --job-name temp_xgb_round1
#SBATCH --nodes 1
#SBATCH --output temp_xgb_round1.out
#SBATCH --error temp_xgb_round1.err
#SBATCH -p high
#SBATCH --cpus-per-task 15
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user butlerj@berkeley.edu
export OMP_NUM_THREADS=1
cd ..
conda run -n antarctic_ars python cross_validation_xgb.py --x_cols max_ocean_SLP_gradient max_landfalling_v850hPa avg_landfalling_minomega max_IWV_ais cumulative_landfalling_area max_south_extent --y_col max_T2M_anomaly_ais --hyperparam_json temp_round1_hyperparams.json --chunk_size 50  --save_name temp_xgb_round1.csv
