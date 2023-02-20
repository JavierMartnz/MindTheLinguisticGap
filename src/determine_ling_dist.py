import os
import argparse
from utils.util import save_gzip
import pandas as pd


def main(params):
    output_path = params.output_path
    signbank_csv = params.signbank_csv
    cngt_root = params.cngt_root

    assert os.path.exists(signbank_csv), f"The indicated file {signbank_csv} does not exist."

    video_signs = [filename for filename in os.listdir(cngt_root) if filename.endswith(".mpg")]

    gloss_list = []
    gloss_id_list = []
    for video in video_signs:
        gloss_list.append(video.split('_')[-2])
        gloss_id_list.append(int(video.split('_')[-1].split('.')[0]))

    gloss_id_freq = {}

    for i in range(len(gloss_list)):
        gloss_id_freq[gloss_id_list[i]] = gloss_id_freq.get(gloss_id_list[i], 0) + 1

    ordered_id_freq = {k: v for k, v in sorted(gloss_id_freq.items(), key=lambda item: item[1], reverse=True)}

    min_gloss_cnt = 100

    filtered_id_freq = {k: v for k, v in ordered_id_freq.items() if v > min_gloss_cnt}

    sb_df = pd.read_csv(signbank_csv)

    ling_df = sb_df[["Signbank ID",
                     "Handedness",
                     "Strong Hand",
                     "Weak Hand",
                     "Handshape Change",
                     "Relation between Articulators",
                     "Location",
                     "Relative Orientation: Movement",
                     "Relative Orientation: Location",
                     "Orientation Change",
                     "Contact Type",
                     "Movement Shape",
                     "Movement Direction",
                     "Repeated Movement",
                     "Alternating Movement"]]

    # these signs have at least 100 clips
    for gloss_id in filtered_id_freq.keys():
        hi_df = ling_df.apply(axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cngt_root",
        type=str,
        default="D:/Thesis/datasets/cngt_single_signs"
    )

    parser.add_argument(
        "--signbank_csv",
        type=str,
        default="D:/Thesis/dictionary-export.csv",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="D:/Thesis/datasets/signbank_vocab.gzip",
    )

    params, _ = parser.parse_known_args()

    main(params)
