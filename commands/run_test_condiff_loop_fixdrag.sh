#!/bin/bash

testname=$1
ckptname=$2

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/craftsman:$(pwd)/condition_diffusion"

ckpt_path="outputs/Diffusion-Conditional/$testname/ckpt/$ckptname.ckpt"
CONFIG_FILE="configs/diffusion/Diffusion-cond-test-fixdrag.yaml"

python launch_condiff.py \
    --config "$CONFIG_FILE" \
    --test \
    --test_batches 559 \
    --num_gpus 1
