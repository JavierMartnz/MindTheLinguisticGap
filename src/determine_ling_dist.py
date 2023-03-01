import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import time

def main(params):
    signbank_csv = params.signbank_csv
    cngt_root = params.cngt_root

    assert os.path.exists(signbank_csv), f"The indicated file {signbank_csv} does not exist."

    # GET DICTIONARY WITH GLOSS COUNT IN CNGT
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

    # we don't want glosses with less than 100 videos to train
    min_gloss_cnt = 100

    filtered_id_freq = {k: v for k, v in ordered_id_freq.items() if v > min_gloss_cnt}

    # GET LINGUISTIC DISTANCE MATRIX (MAINLY NATALIE HOLLAIN'S CODE)
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

    # keep only the rows with the filtered glosses
    ling_df = ling_df[ling_df["Signbank ID"].isin(list(filtered_id_freq.keys()))]

    # get dictionaries to work with id and glosses indistinctively
    id_to_gloss = pd.Series(sb_df["Annotation ID Gloss (Dutch)"].values, index=sb_df["Signbank ID"]).to_dict()
    gloss_to_id = pd.Series(sb_df["Signbank ID"].values, index=sb_df["Annotation ID Gloss (Dutch)"]).to_dict()

    # get the ids of the filtered dataframe and remove them from the df
    gloss_ids = ling_df["Signbank ID"].values.tolist()
    ling_df = ling_df.drop(columns=["Signbank ID"])

    # fill NaN values with -1 for easier comparison
    ling_df = ling_df.fillna(-1)

    ling_np = ling_df.to_numpy()
    num = len(gloss_ids)

    # Linguistic difference matrix
    ling_dist = np.zeros((num, num))

    # Loop over the indices of the glosses, and then compare the glosses
    for i in tqdm(range(num-1)):
        for j in range(i+1, num):
            gloss1, gloss2 = ling_np[i], ling_np[j]
            # use this loop instead of np.where() since it's faster
            dist = 0
            for k in range(len(gloss1)):
                dist += 1 if gloss1[k] != gloss2[k] else 0

            ling_dist[i, j] = dist
            # Distance should be symmetrical
            ling_dist[j, i] = ling_dist[i, j]

    # now we want to get a dictionary with the frequency of the clips and their linguistic distance
    distances_and_freqs = {}
    for gloss_id_idx in range(len(gloss_ids)):
        most_occ_glosses = {}
        for i in range(1, 11):
            matches = np.where(ling_dist[gloss_id_idx, :] == i)[0]
            if len(matches) > 0:
                idx_matches = []
                for match in matches:
                    idx_matches.append(list(filtered_id_freq.keys()).index(gloss_ids[match]))
                # since the ids are ordered based on frequency, then the smallest index is the gloss with more clips
                gloss_id = list(filtered_id_freq.keys())[min(idx_matches)]
                most_occ_glosses[i] = (gloss_id, filtered_id_freq[gloss_id])

        distances_and_freqs[gloss_ids[gloss_id_idx]] = (filtered_id_freq[gloss_ids[gloss_id_idx]], most_occ_glosses)

    # order the dictionary based on the frequency of the glosses
    ordered_distances_and_freqs = {k: v for k, v in sorted(distances_and_freqs.items(), key=lambda item: item[1][0], reverse=True)}

    for gloss_id in ordered_distances_and_freqs.keys():

        # first check if they match in length
        if len(ordered_distances_and_freqs[gloss_id][1].keys()) == len(np.arange(1, 11)) and np.equal(np.array(list(ordered_distances_and_freqs[gloss_id][1].keys())), np.arange(1, 11)).all():
            print(f"\n{id_to_gloss[gloss_id]} ({ordered_distances_and_freqs[gloss_id][0]} clips) has signs with ling dist:")
            for key in ordered_distances_and_freqs[gloss_id][1]:
                print(f"-{key}: {id_to_gloss[ordered_distances_and_freqs[gloss_id][1][key][0]]} with {ordered_distances_and_freqs[gloss_id][1][key][1]} clips")


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
