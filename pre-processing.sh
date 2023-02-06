#!/usr/bin/env bash

source /vol/tensusers5/jmartinez/pretraini3d/bin/activate
cd /vol/tensusers5/jmartinez/MindTheLinguisticGap
python3 ./src/get_signbank_vocab_from_csv.py --signbank_csv=/vol/tensusers5/jmartinez/dictionary-export.csv --output_path=/vol/tensusers5/jmartinez/datasets/signbank_vocab.gzip
python3 ./src/split_and_resize.py --root=/vol/tensusers5/jmartinez/datasets --cngt_folder=CNGT_complete --cngt_output_folder=CNGT_isolated_signers_512 --sb_folder=NGT_Signbank --sb_output_folder=NGT_Signbank_512 --video_size=512 --framerate=25 --window_size=16
python3 ./src/trim_signs_cngt.py --root=/vol/tensusers5/jmartinez/datasets --cngt_folder=CNGT_final_512res --cngt_output_folder=cngt_single_signs_512 --signbank_vocab_file=signbank_vocab.gzip --window_size=16
deactivate
