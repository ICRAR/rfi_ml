#!/bin/bash --login
# SLURM directives
#
#SBATCH --job-name=sam_plots
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=pleiades
#SBATCH --mem=32000M

/home/sfoster/env/bin/python /home/sfoster/rfi_ml/src/plots/plots_01.py /home/sfoster/rfi_ml/data/v255ae_Mp_072_060000.lba ./mp_plots/
