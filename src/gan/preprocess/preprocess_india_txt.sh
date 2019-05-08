#!/usr/bin/env bash

PYTHON="../../../venv/bin/python"
DATA_PATH="../../../data/power_line_RFI_data"
OUTPUT_PATH="$DATA_PATH/preprocessed"
SAMPLE_RATE="200000000"

mkdir -p ${OUTPUT_PATH}

${PYTHON} main.py ${DATA_PATH}/C148700000.txt ${OUTPUT_PATH}/C148700000.hdf5 --sample_rate ${SAMPLE_RATE}
${PYTHON} main.py ${DATA_PATH}/C148700001.txt ${OUTPUT_PATH}/C148700001.hdf5 --sample_rate ${SAMPLE_RATE}
${PYTHON} main.py ${DATA_PATH}/C148700002.txt ${OUTPUT_PATH}/C148700002.hdf5 --sample_rate ${SAMPLE_RATE}