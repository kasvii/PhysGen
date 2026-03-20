#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

python launch_finetuneall.py \
    --config configs/finetune_all/MultiTaskJoint.yaml \
    --task joint \
    --gpu 0,1,2,3
