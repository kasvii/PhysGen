#!/bin/bash

# Training parameters
CONFIG_FILE="configs/diffusion/Diffusion-uncond-train.yaml"
GPUS=4

python launch_uncondiff.py \
    --config "$CONFIG_FILE" \
    --num_gpus $GPUS 