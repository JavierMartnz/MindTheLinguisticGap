#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap/src
python train_i3d.py --config_path=/vol/tensusers5/jmartinez/MindTheLinguisticGap/config.yaml
deactivate
