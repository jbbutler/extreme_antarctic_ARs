#!/bin/bash
#SBATCH --job-name snow_gbex_round1
#SBATCH --nodes 1
#SBATCH --output ../../../../outputs/logs/snow_gbex_round1.out
#SBATCH --error ../../../../outputs/logs/snow_gbex_round1.err
#SBATCH -p high
#SBATCH --cpus-per-task 15
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user butlerj@berkeley.edu
export OMP_NUM_THREADS=1
cd ..
conda run -n extreme_antarctic_ars --no-capture-output Rscript extreme_cv_gbex.R --x_cols max_ocean_SLP_gradient max_landfalling_v850hPa avg_landfalling_minomega max_IWV_ais cumulative_landfalling_area max_south_extent --y_col cumulative_snowfall_ais --hyperparam_json snow_gbex_round1_hyperparams.json --grf_hyperparam_json best_snow_grf.json --chunk_size 20 --ncores 15  --save_name snow_gbex_round1.csv
