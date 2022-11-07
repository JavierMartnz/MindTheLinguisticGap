import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd


def main():
    isolated_signs_root = "D:/Thesis/datasets/cngt_single_signs/"
    minimal_pairs_csv = "D:/Thesis/dictionary-export-minimalpairs.csv"

    mp_df = pd.read_csv(minimal_pairs_csv)

    id_list = []
    gloss_list = []

    video_signs = [filename for filename in os.listdir(isolated_signs_root) if filename.endswith(".mpg")]

    for video in video_signs:
        gloss = video.split('_')[-2]
        id = int(video.split('_')[-1].split('.')[0])
        gloss_list.append(gloss)
        id_list.append(id)

    print(f"There are {len(gloss_list)} glosses, {len(set(gloss_list))} unique.")

    gloss_freq = {}
    id_freq = {}

    for i in range(len(gloss_list)):
        gloss_freq[gloss_list[i]] = gloss_freq.get(gloss_list[i], 0) + 1
        id_freq[id_list[i]] = id_freq.get(id_list[i], 0) + 1

    ordered_gloss_freq = {k: v for k, v in sorted(gloss_freq.items(), key=lambda item: item[1], reverse=True)}
    ordered_id_freq = {k: v for k, v in sorted(id_freq.items(), key=lambda item: item[1], reverse=True)}

    min_gloss_cnt = 50

    filtered_gloss_freq = {k: v for k, v in ordered_gloss_freq.items() if v > min_gloss_cnt}
    filtered_id_freq = {k: v for k, v in ordered_id_freq.items() if v > min_gloss_cnt}

    # print(f"The top {min_gloss_cnt} glosses are:\n{filtered_gloss_freq}")

    minimal_pairs_dict = {}

    for gloss_id in filtered_id_freq.keys():
        if not mp_df.loc[mp_df["ID"] == int(gloss_id)]["ID.1"].isnull().any():
            minimal_pairs_id = list(map(int, mp_df.loc[mp_df["ID"] == int(gloss_id)]["ID.1"]))
            mp_dict = {mp_id: ordered_id_freq[mp_id] if mp_id in id_list else 0 for mp_id in minimal_pairs_id}
            minimal_pairs_dict[gloss_id] = {k: v for k, v in sorted(mp_dict.items(), key=lambda item: item[1], reverse=True)}

    for gloss_id in minimal_pairs_dict.keys():
        mps = ""
        for gloss_id_mp in minimal_pairs_dict[gloss_id].keys():
            if gloss_id_mp in id_list and minimal_pairs_dict[gloss_id][gloss_id_mp] > min_gloss_cnt:
                if 0.45 <= ordered_id_freq[gloss_id]/(ordered_id_freq[gloss_id] + minimal_pairs_dict[gloss_id][gloss_id_mp]) <= 0.55:
                    mps += f"{gloss_list[id_list.index(gloss_id_mp)]}: {minimal_pairs_dict[gloss_id][gloss_id_mp]}; "

        if len(mps) > 0:
            print(f"{gloss_list[id_list.index(gloss_id)]} with {ordered_id_freq[gloss_id]} occurrences has as minimal pairs:")
            print(f"\t{mps}", end="")
            print("\n")

    fig = go.Figure(go.Bar(
        x=list(filtered_gloss_freq.values()),
        y=list(filtered_gloss_freq.keys()),
        orientation='h'))

    fig.show()


if __name__ == "__main__":
    main()
