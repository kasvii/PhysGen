#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

python launch_physdec.py --config ./configs/physdec/PhysDec-train.yaml \
                          --train --gpu 0,1,2,3