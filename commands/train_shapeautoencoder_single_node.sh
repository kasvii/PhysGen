#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

python launch_shapevae.py --config ./configs/shape-autoencoder/Dora-VAE-train.yaml \
                          --train --gpu 0,1,2,3