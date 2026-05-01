#!/bin/bash
#SBATCH --job-name temp_grf_round2
#SBATCH --nodes 1
#SBATCH --output ../../../../outputs/logs/temp_grf_round2.out
#SBATCH --error ../../../../outputs/logs/temp_grf_round2.err
#SBATCH -p high
#SBATCH --cpus-per-task 15
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user butlerj@berkeley.edu
export OMP_NUM_THREADS=1
cd ..
conda run -n extreme_antarctic_ars --no-capture-output Rscript intermediate_cv_gbex.R --x_cols max_ocean_SLP_gradient max_landfalling_v850hPa avg_landfalling_minomega max_IWV_ais cumulative_landfalling_area max_south_extent --y_col max_T2M_anomaly_ais --hyperparam_json temp_grf_round2_hyperparams.json --chunk_size 20 --ncores 15  --save_name temp_grf_round2.csv
