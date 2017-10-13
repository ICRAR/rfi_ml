#!/bin/bash --login
# SLURM directives
#
#SBATCH --job-name=gmrt_cnn
#SBATCH --account=pawsey0245
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --mail-user kevin.vinsen@icrar.org

module load use.own
module load broadwell gcc/5.4.0 cuda python magma cffi

source /group/pawsey0245/kvinsen/pytorch/bin/activate

cd /group/pawsey0245/kvinsen/rfi_ml/src
# srun -n 1 python train_gmrt_cnn.py --use-gpu --save gmrt_cnn.model.saved
srun -n 1 python -m cProfile -o train_gmrt_cnn.prof train_gmrt_cnn.py --use-gpu --save gmrt_cnn.model.saved
