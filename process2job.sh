#!/bin/bash

# Define variables
OUT_DIR="out/"
DATA_DIR="/media/eric/D/datasets/"

python train.py "$@" --out_dir="$OUT_DIR" --data_dir="$DATA_DIR"