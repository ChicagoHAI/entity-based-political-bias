#!/bin/bash

PROJ_DIR="~/entity-based-political-bias"  # replace with your project path
FEXT="all"

python get_results.py \
    --proj_dir $PROJ_DIR \
    --fext $FEXT