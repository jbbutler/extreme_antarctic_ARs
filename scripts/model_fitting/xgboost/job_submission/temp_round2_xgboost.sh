#!/bin/bash
#SBATCH --job-name temp_xgb_round2
#SBATCH --nodes 1
#SBATCH --output ../../../../outputs/logs/temp_xgb_round2.out
#SBATCH --error ../../../../outputs/logs/temp_xgb_round2.err
#SBATCH -p high
#SBATCH --cpus-per-task 15
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user butlerj@berkeley.edu
export OMP_NUM_THREADS=1
cd ..
conda run -n extreme_antarctic_ars --no-capture-output python cross_validation_xgb.py --x_cols max_ocean_SLP_gradient max_landfalling_v850hPa avg_landfalling_minomega max_IWV_ais cumulative_landfalling_area max_south_extent --y_col max_T2M_anomaly_ais --hyperparam_json temp_round2.json --chunk_size 20  --save_name temp_xgb_round2.csv --shrink
