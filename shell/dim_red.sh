#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/dim_red.py --config_path=/vol/tensusers5/jmartinez/MindTheLinguisticGap/dim_red_config.yaml --fig_output_root=/vol/tensusers5/jmartinez/graphs
deactivate