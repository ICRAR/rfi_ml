#!/bin/bash --login
# SLURM directives
#
#SBATCH --job-name=sam_find_rfi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=pleiades
#SBATCH --mem=4000M

/home/sfoster/env/bin/python /home/sfoster/rfi_ml/src/gan/preprocess.py /home/sfoster/rfi_ml/data/v255ae_At_072_060000.lba at_data_2048.hdf5 --max_ffts 100
/home/sfoster/env/bin/python /home/sfoster/rfi_ml/src/gan/preprocess.py /home/sfoster/rfi_ml/data/vt255ae_Pa_072_060000.lba pa_data_2048.hdf5 --max_ffts 100
/home/sfoster/env/bin/python /home/sfoster/rfi_ml/src/gan/preprocess.py /home/sfoster/rfi_ml/data/v255ae_Mp_072_060000.lba mp_data_2048.hdf5 --max_ffts 100