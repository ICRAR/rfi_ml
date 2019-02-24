#!/bin/bash --login
# SLURM directives
#
#SBATCH --job-name=my_job
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mlgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000

python job.py

