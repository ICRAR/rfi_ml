#!/bin/bash --login
# SLURM directives
#
#SBATCH --job-name=sam_find_rfi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=pleiades
#SBATCH --mem=128000M

/home/sfoster/env/bin/python /home/sfoster/rfi_ml/src/find_rfi.py /home/sam/Projects/rfi_ml/data/v255ae_Mp_072_060000.lba found_peak 262144
