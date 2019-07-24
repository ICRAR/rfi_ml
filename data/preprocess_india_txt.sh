#!/usr/bin/env bash

PYTHON="../venv/bin/python"
PREPROCESS="../src/gan/preprocess/main.py"
PREPROCESS_FFT="../src/gan/preprocess/fft/main.py"
DATA_PATH="./raw/power_line_RFI_data"
OUTPUT_PATH="./processed"
SAMPLE_RATE="200000000"

mkdir -p ${OUTPUT_PATH}

${PYTHON} ${PREPROCESS} ${DATA_PATH}/C148700000.txt ${OUTPUT_PATH}/C148700000.hdf5 --sample_rate ${SAMPLE_RATE}
${PYTHON} ${PREPROCESS} ${DATA_PATH}/C148700001.txt ${OUTPUT_PATH}/C148700001.hdf5 --sample_rate ${SAMPLE_RATE}
${PYTHON} ${PREPROCESS} ${DATA_PATH}/C148700002.txt ${OUTPUT_PATH}/C148700002.hdf5 --sample_rate ${SAMPLE_RATE}

${PYTHON} ${PREPROCESS_FFT} ${OUTPUT_PATH}/C148700000.hdf5 ${OUTPUT_PATH}/C148700000_fft.hdf5
${PYTHON} ${PREPROCESS_FFT} ${OUTPUT_PATH}/C148700001.hdf5 ${OUTPUT_PATH}/C148700001_fft.hdf5
${PYTHON} ${PREPROCESS_FFT} ${OUTPUT_PATH}/C148700002.hdf5 ${OUTPUT_PATH}/C148700002_fft.hdf5