#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/utils/determine_ling_dist.py --cngt_root=/vol/tensusers5/jmartinez/datasets/cngt_single_signs_256 --signbank_csv=/vol/tensusers5/jmartinez/datasets/dictionary-export.csv --txtfile_output_root=/vol/tensusers5/jmartinez
