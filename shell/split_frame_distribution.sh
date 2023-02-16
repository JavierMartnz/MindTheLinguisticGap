#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/get_split_frame_distributions.py --root=/vol/tensusers5/jmartinez/datasets --cngt_folder=cngt_single_signs_12fps --sb_folder=NGT_Signbank_12fps --sb_vocab_file=signbank_vocab.gzip --fig_output_root=/vol/tensusers5/jmartinez/graphs
deactivate
