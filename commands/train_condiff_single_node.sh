#!/bin/bash

GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
CONFIG_FILE="configs/diffusion/Diffusion-cond-train.yaml"

python launch_condiff.py \
    --config "$CONFIG_FILE" \
    --num_gpus $GPUS 