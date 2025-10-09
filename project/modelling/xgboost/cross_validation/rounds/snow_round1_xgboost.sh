#!/bin/bash
#SBATCH --job-name=xgb_snow_cv
#SBATCH --mail-type=ALL
#SBATCH --mail-user=butlerj@berkeley.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH -o cv.out #File to which standard out will be written
#SBATCH -e cv.err #File to which standard err will be written

cd ..
module load conda
conda activate antarctic_ars

python cross_validation.py --x_cols max_ocean_SLP_gradient max_landfalling_v850hPa max_landfalling_omega500 max_IWV_ais cumulative_landfalling_area max_south_extent --y_col cumulative_snowfall_ais --hyperparam_json snow_round1_hyperparams.json --save_name snow_round1_cv_res.csv