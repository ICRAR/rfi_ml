#!/usr/bin/env bash

PYTHON="python -m"
PREPROCESS="src.preprocess.main"
PREPROCESS_FFT="src.preprocess.fft.main"
DATA_PATH="data/raw/power_line_RFI_data"
OUTPUT_PATH="data/processed"
SAMPLE_RATE="200000000"

if [[ ! -d ${DATA_PATH} ]]; then
    echo "${DATA_PATH} does not exist"
fi

mkdir -p ${OUTPUT_PATH}

source venv/bin/activate

${PYTHON} ${PREPROCESS} ${DATA_PATH}/C148700000.txt ${OUTPUT_PATH}/C148700000.hdf5 --sample_rate ${SAMPLE_RATE}
${PYTHON} ${PREPROCESS} ${DATA_PATH}/C148700001.txt ${OUTPUT_PATH}/C148700001.hdf5 --sample_rate ${SAMPLE_RATE}
${PYTHON} ${PREPROCESS} ${DATA_PATH}/C148700002.txt ${OUTPUT_PATH}/C148700002.hdf5 --sample_rate ${SAMPLE_RATE}

${PYTHON} ${PREPROCESS_FFT} ${OUTPUT_PATH}/C148700000.hdf5 ${OUTPUT_PATH}/C148700000_fft.hdf5
${PYTHON} ${PREPROCESS_FFT} ${OUTPUT_PATH}/C148700001.hdf5 ${OUTPUT_PATH}/C148700001_fft.hdf5
${PYTHON} ${PREPROCESS_FFT} ${OUTPUT_PATH}/C148700002.hdf5 ${OUTPUT_PATH}/C148700002_fft.hdf5