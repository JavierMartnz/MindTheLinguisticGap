#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/gloss_freq_cngt_clips.py --cngt_root=/vol/tensusers5/jmartinez/datasets/cngt_single_signs_512 --fig_output_root=/vol/tensusers5/jmartinez/graphs --max_glosses=10
deactivate
