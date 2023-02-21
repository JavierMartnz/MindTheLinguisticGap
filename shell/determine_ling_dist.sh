#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/determine_ling_dist.py --cngt_root=/vol/tensusers5/jmartinez/datasets/cngt_single_signs --signbank_csv=/vol/tensusers5/jmartinez/datasets/dictionary-export.csv
