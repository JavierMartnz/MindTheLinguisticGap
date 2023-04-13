#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/get_intrinsic_dim.py --config_path=/vol/tensusers5/jmartinez/MindTheLinguisticGap/get_id_config.yaml
deactivate