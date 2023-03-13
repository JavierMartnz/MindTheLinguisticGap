import argparse
import os
import pandas as pd

def main(params):
    cngt_root = params.cngt_root
    mp_csv_path = params.mp_csv_path

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

    min_gloss_cnt = 50

    filtered_id_freq = {k: v for k, v in ordered_id_freq.items() if v > min_gloss_cnt}

    # read csv where the minimal pairs are stored
    mp_df = pd.read_csv(mp_csv_path)

    minimal_pairs_dict = {}

    for gloss_id in filtered_id_freq.keys():
        if not mp_df.loc[mp_df["ID"] == int(gloss_id)]["ID.1"].isnull().any():
            minimal_pairs_id = list(map(int, mp_df.loc[mp_df["ID"] == int(gloss_id)]["ID.1"]))
            mp_dict = {mp_id: ordered_id_freq[mp_id] if mp_id in gloss_id_list else 0 for mp_id in minimal_pairs_id}
            minimal_pairs_dict[gloss_id] = {k: v for k, v in
                                            sorted(mp_dict.items(), key=lambda item: item[1], reverse=True)}

    for gloss_id in minimal_pairs_dict.keys():
        mps = ""
        for gloss_id_mp in minimal_pairs_dict[gloss_id].keys():
            if gloss_id_mp in gloss_id_list and minimal_pairs_dict[gloss_id][gloss_id_mp] > min_gloss_cnt:
                if 0.45 <= ordered_id_freq[gloss_id] / (
                        ordered_id_freq[gloss_id] + minimal_pairs_dict[gloss_id][gloss_id_mp]) <= 0.55:
                    mps += f"{gloss_list[gloss_id_list.index(gloss_id_mp)]}: {minimal_pairs_dict[gloss_id][gloss_id_mp]};\t"

        if len(mps) > 0:
            print(
                f"{gloss_list[gloss_id_list.index(gloss_id)]} with {ordered_id_freq[gloss_id]} occurrences has as minimal pairs:")
            print(f"\t{mps}", end="")
            print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cngt_root",
        type=str,
        default="D:/Thesis/datasets/cngt_single_signs"
    )

    parser.add_argument(
        "--mp_csv_path",
        type=str,
        default="D:/Thesis/dictionary-export-minimalpairs.csv"
    )

    params, _ = parser.parse_known_args()
    main(params)
