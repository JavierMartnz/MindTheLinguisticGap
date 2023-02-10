import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def main(params):
    cngt_root = params.cngt_root
    fig_output_root = params.fig_output_root
    max_glosses = params.max_glosses

    gloss_id_list = []
    gloss_list = []

    video_signs = [filename for filename in os.listdir(cngt_root) if filename.endswith(".mpg")]

    for video in video_signs:
        gloss_list.append(video.split('_')[-2])
        gloss_id_list.append(int(video.split('_')[-1].split('.')[0]))

    print(f"There are {len(gloss_list)} glosses, {len(set(gloss_list))} unique.")

    gloss_freq = {}
    gloss_id_freq = {}

    for i in range(len(gloss_list)):
        gloss_freq[gloss_list[i]] = gloss_freq.get(gloss_list[i], 0) + 1
        gloss_id_freq[gloss_id_list[i]] = gloss_id_freq.get(gloss_id_list[i], 0) + 1

    ordered_gloss_freq = {k: v for k, v in sorted(gloss_freq.items(), key=lambda item: item[1], reverse=True)}
    ordered_id_freq = {k: v for k, v in sorted(gloss_id_freq.items(), key=lambda item: item[1], reverse=True)}

    min_gloss_cnt = 50

    filtered_gloss_freq = {k: v for k, v in ordered_gloss_freq.items() if v > min_gloss_cnt}
    filtered_id_freq = {k: v for k, v in ordered_id_freq.items() if v > min_gloss_cnt}

    glosses = list(filtered_gloss_freq.keys())
    glosses_freq = list(filtered_gloss_freq.values())

    colors = sns.color_palette('pastel')

    if max_glosses != -1:
        for key in list(filtered_gloss_freq.keys())[:max_glosses]:
            print(f"{key}: {filtered_gloss_freq[key]}")

        glosses = glosses[:max_glosses]
        glosses_freq = glosses_freq[:max_glosses]
        colors = colors[:max_glosses]

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    plt.barh(glosses, glosses_freq, color=colors)
    plt.gca().invert_yaxis()
    plt.title("Number of clips per gloss in the processed CNGT")
    plt.xlabel("Clips")
    plt.ylabel("Gloss")
    plt.tight_layout()

    os.makedirs(fig_output_root, exist_ok=True)

    plt.savefig(os.path.join(fig_output_root, "gloss_freq_cngt_clips"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cngt_root",
        type=str,
        default="D:/Thesis/datasets/cngt_single_signs"
    )

    parser.add_argument(
        "--fig_output_root",
        type=str,
        default="D:/Thesis/graphs"
    )

    parser.add_argument(
        "--max_glosses",
        type=int,
        default="-1"
    )

    params, _ = parser.parse_known_args()
    main(params)
