#!/usr/bin/env bash

PYTHON="python -m"
PREPROCESS="src.preprocess.main"
PREPROCESS_FFT="src.preprocess.fft.main"
DATA_PATH="data/raw/vlba"
OUTPUT_PATH="data/processed"

if [[ ! -d ${DATA_PATH} ]]; then
    echo "${DATA_PATH} does not exist"
fi


mkdir -p ${OUTPUT_PATH}

source venv/bin/activate

# Sample rate set to 1000000 for testing. remove to process entire file.
COMMON_ARGS="--sample_rate 32000000 --max_samples 1000000 --lba_obs_file data/raw/vlba/v255ae.vex"

${PYTHON} ${PREPROCESS} ${DATA_PATH}/v255ae_At_072_060000.lba ${OUTPUT_PATH}/v255ae_At_072_060000.hdf5 ${COMMON_ARGS} --lba_antenna_name At
${PYTHON} ${PREPROCESS} ${DATA_PATH}/v255ae_Mp_072_060000.lba ${OUTPUT_PATH}/v255ae_Mp_072_060000.hdf5 ${COMMON_ARGS} --lba_antenna_name Mp
${PYTHON} ${PREPROCESS} ${DATA_PATH}/vt255ae_Pa_072_060000.lba ${OUTPUT_PATH}/vt255ae_Pa_072_060000.hdf5 ${COMMON_ARGS} --lba_antenna_name Pa

${PYTHON} ${PREPROCESS_FFT} ${OUTPUT_PATH}/v255ae_At_072_060000.hdf5 ${OUTPUT_PATH}/v255ae_At_072_060000_fft.hdf5
${PYTHON} ${PREPROCESS_FFT} ${OUTPUT_PATH}/v255ae_Mp_072_060000.hdf5 ${OUTPUT_PATH}/v255ae_Mp_072_060000_fft.hdf5
${PYTHON} ${PREPROCESS_FFT} ${OUTPUT_PATH}/vt255ae_Pa_072_060000.hdf5 ${OUTPUT_PATH}/vt255ae_Pa_072_060000_fft.hdf5