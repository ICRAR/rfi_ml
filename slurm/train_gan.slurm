#!/bin/bash --login
# SLURM directives
#
#SBATCH --job-name=train_gan
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mlgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH -o /home/ipromnorakit/log/train_gan_%j.err
#SBATCH -e /home/ipromnorakit/log/train_gan_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user ittp.michael@gmail.com

module load /home/ipromnorakit/modulefile/cuda-9.0 

source ~/.profile
conda activate rfi_ml

cd /home/ipromnorakit/rfi_ml/src/gan
srun -n 1 python ./train.py 
