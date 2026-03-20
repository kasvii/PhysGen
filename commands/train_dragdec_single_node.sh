#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

python launch_dragdec.py --config ./configs/dragdec/DragDec-train.yaml \
                          --train --gpu 0