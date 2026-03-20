#!/bin/bash

python launch_uncondiff.py \
    --config configs/diffusion/Diffusion-uncond-test-givenshape.yaml \
    --test \
    --test_batches 1147 \
    --num_gpus 1 \
    --seed 42