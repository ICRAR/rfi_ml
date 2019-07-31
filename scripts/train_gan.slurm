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
#SBATCH -o /home/sfoster/train_gan_%j.err
#SBATCH -e /home/sfoster/train_gan_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user sam6321@live.com.au

module load /home/sfoster/rfi_ml/modulefile/cuda-9.0

cd /home/sfoster/rfi_ml/src/gan/
/home/sfoster/env/bin/python train.py /home/sfoster/rfi_ml/slurm/gan_config.settings

