#!/bin/bash --login
# SLURM directives
#
#SBATCH --job-name=sam_find_rfi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=pleiades
#SBATCH --mem=4000M

/home/sfoster/env/bin/python /home/sfoster/rfi_ml/src/find_rfi.py /home/sfoster/rfi_ml/data/vt255ae_Pa_072_060000.lba found_peak_pa 262144 1241513984
