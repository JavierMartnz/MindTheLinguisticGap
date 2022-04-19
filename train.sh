#!/usr/bin/env bash

cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
CUDA_LAUNCH_BLOCKING=1 python ./src/train_i3d.py --config_path=/vol/tensusers5/jmartinez/MindTheLinguisticGap/config.yaml
