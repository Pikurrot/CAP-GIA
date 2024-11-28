#!/bin/bash

# Define variables
OUT_DIR="out/"
DATA_DIR="/media/eric/D/datasets/"

# Ensure an argument is passed
if [ "$#" -lt 1 ]; then
	echo "Error: No arguments provided. Usage: $0 --[train|test] --[args...]"
	exit 1
fi

# Get the first argument
MODE="$1"
shift # Remove the first argument from the list

# Check the mode and call the appropriate Python script
if [ "$MODE" == "--train" ]; then
	python train.py "$@" --out_dir="$OUT_DIR" --data_dir="$DATA_DIR"
elif [ "$MODE" == "--test" ]; then
	python test.py "$@" --out_dir="$OUT_DIR" --data_dir="$DATA_DIR"
else
	echo "Error: Invalid mode '$MODE'. Use '--train' or '--test'."
	exit 1
fi
