#!/bin/bash --login
# SLURM directives
#
#SBATCH --job-name=sam_preprocess_v255ae_Mp_072_060000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=pleiades
#SBATCH --mem=4000M

/home/sfoster/env/bin/python /home/sfoster/rfi_ml/src/gan/preprocess.py /home/sfoster/rfi_ml/data/v255ae_Mp_072_060000.lba mp_data_2048.hdf5
