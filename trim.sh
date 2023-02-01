#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/trim_signs_cngt.py --root=/vol/tensusers5/jmartinez/datasets --dataset_root=CNGT_final_512res --signbank_vocab_file=signbank_vocab.gzip --output_root=cngt_single_signs_512 --window_size=16
deactivate
