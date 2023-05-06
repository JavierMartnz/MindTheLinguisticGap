#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/test_i3d.py --config_path=/vol/tensusers5/jmartinez/MindTheLinguisticGap/config/predict_config.yaml
deactivate
