#!/bin/bash
# Render dataset samples (grey mesh + ground-truth surface pressure) in the same format as the
# SP-VAE test figures. Usage: bash commands/vis_dataset_sample.sh <uid> [<uid> ...]

python vis_dataset_sample.py \
    --uid "$@" \
    --data_root data/drivaernet_plus \
    --out_dir outputs/vis_dataset
