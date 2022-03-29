#!/bin/bash

MAIN=/data/home/rndkoa/2020
CONFIG=${MAIN}/DNN_MSRN/TRAIN/MSRN/DABA/config.json
SRCS=${MAIN}/DNN_MSRN/TRAIN/MSRN/SRCS
PYTHON=${MAIN}/envs/pytorch_cpu/bin/python3
export PYTHONPATH=${PYTHON}:${SRCS}
export ECCODES_DEFINITION_PATH=${MAIN}/envs/pytorch_cpu/share/eccodes/definitions


${PYTHON} ${SRCS}/main.py --config ${CONFIG}

