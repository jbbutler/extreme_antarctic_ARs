#!/bin/bash
#SBATCH --job-name test_cv
#SBATCH --nodes 1
#SBATCH --output test_cv%j.out
#SBATCH --error test_cv%j.err
#SBATCH -p high
#SBATCH --cpus-per-task 3
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user butlerj@berkeley.edu
export OMP_NUM_THREADS=1
cd ..
conda run -n antarctic_ars python cross_validation.py --x_cols max_ocean_SLP_gradient max_landfalling_v850hPa max_landfalling_omega500 max_IWV_ais cumulative_landfalling_area max_south_extent --y_col cumulative_snowfall_ais --hyperparam_json test_hyper.json --chunk_size 16  --save_name test_res.csv
