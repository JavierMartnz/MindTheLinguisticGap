#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/get_minimal_pairs.py cngt_root=/vol/tensusers5/jmartinez/cngt_single_signs_512 --mp_csv_path=/vol/tensusers5/jmartinez/datasets/dictionary-export-minimalpairs.csv
deactivate
