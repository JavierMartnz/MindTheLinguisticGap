#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/autoencoder.py --config_path=/vol/tensusers5/jmartinez/MindTheLinguisticGap/autoencoder_config.yaml
