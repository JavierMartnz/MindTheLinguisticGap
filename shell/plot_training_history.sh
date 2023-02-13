#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/plot_training_history.py --trained_models_parent=/vol/tensusers5/jmartinez/models/i3d --fig_output_root=/vol/tensusers5/jmartinez/graphs --config_path=/vol/tensusers5/jmartinez/MindTheLinguisticGap/config.yaml
deactivate
