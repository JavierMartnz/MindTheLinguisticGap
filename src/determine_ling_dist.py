import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

def main(params):
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

    id_to_gloss = pd.Series(sb_df["Annotation ID Gloss (Dutch)"].values, index=sb_df["Signbank ID"]).to_dict()

    all_diffs = {}

    # these signs have at least 100 clips
    for gloss_id in tqdm(filtered_id_freq.keys()):
        gloss_id_values = ling_df[ling_df["Signbank ID"] == gloss_id].values.tolist()[0]
        gloss_id_values.remove(gloss_id)

        diff_cnt = {}

        other_glosses = list(filtered_id_freq.keys())
        other_glosses.remove(gloss_id)

        for other_id in other_glosses:
            other_id_values = ling_df[ling_df["Signbank ID"] == other_id].values.tolist()[0]
            other_id_values.remove(other_id)

            ling_diff = sum(
                1 for i, j in zip(gloss_id_values, other_id_values) if i != j and not (pd.isnull(i) and pd.isnull(j)))

            diff_cnt[id_to_gloss[other_id]] = (ling_diff, filtered_id_freq[other_id])
            ordered_diff_cnt = {k: v for k, v in sorted(diff_cnt.items(), key=lambda item: item[1][0])}

        all_diffs[id_to_gloss[gloss_id]] = (filtered_id_freq[gloss_id], ordered_diff_cnt)

    for gloss in list(all_diffs.keys()):

        ling_dists = {}

        for key in all_diffs[gloss][1].keys():

            ling_dist = all_diffs[gloss][1][key][0]
            numb_occ = all_diffs[gloss][1][key][1]

            if ling_dist not in ling_dists.keys():
                ling_dists[ling_dist] = numb_occ

        all_ling_dists = np.array(list(ling_dists.keys()))[:10] if len(ling_dists.keys()) > 10 else np.array(
            list(ling_dists.keys()))
        perf_ling_dists = np.arange(1, 11) if len(ling_dists.keys()) > 10 else np.arange(1, len(ling_dists.keys()) + 1)

        if (all_ling_dists == perf_ling_dists).all():
            ling_dist_values = list(ling_dists.values())
            min_num_samples = min(ling_dist_values)
            min_sample = list(all_diffs[gloss][1].keys())[list(all_diffs[gloss][1].values()).index(
                (list(ling_dists.keys())[ling_dist_values.index(min_num_samples)], min_num_samples))]

            print(
                f"{gloss} ({all_diffs[gloss][0]}) has signs with ling dist {ling_dists.keys()}.\nThe sign with the minimum number of samples is {min_sample} with {min_num_samples} clips.\n")


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

    params, _ = parser.parse_known_args()

    main(params)
