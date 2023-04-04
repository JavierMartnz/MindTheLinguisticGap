import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from tqdm import tqdm
import argparse
import statistics
from pathlib import Path
import subprocess
import seaborn as sns

from src.utils.util import load_gzip

def print_stats(clip_durations: list, framerate: int, dataset: str, fig_output_root: str):
    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")
    colors = sns.color_palette('pastel')

    upper_quartile = np.percentile(clip_durations, 75)
    lower_quartile = np.percentile(clip_durations, 25)
    iqr = upper_quartile - lower_quartile

    clip_durations = np.array(clip_durations)

    upper_whisker = clip_durations[np.where(clip_durations <= upper_quartile + 1.5 * iqr, True, False)].max()
    lower_whisker = clip_durations[np.where(clip_durations >= lower_quartile - 1.5 * iqr, True, False)].min()

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)})

    ax_box.boxplot(clip_durations, showfliers=False, vert=False)
    ax_box.set_yticks([])

    ax_hist.hist(clip_durations, bins='auto', align='mid', color=colors[0])

    plt.xlim([lower_whisker - 1, upper_whisker + 1])
    plt.gca().set_xticks(np.linspace(lower_whisker - 1, upper_whisker + 1, num=8, dtype=int))
    # plt.suptitle(f"Number of frames per clip in {dataset} at {framerate} fps")
    plt.ylabel("Number of clips")
    plt.xlabel("Number of frames")
    plt.tight_layout()

    os.makedirs(fig_output_root, exist_ok=True)
    plt.savefig(os.path.join(fig_output_root, f"{dataset}_clips_duration_{framerate}fps.png"))

def get_stats_cngt(cngt_root: str, framerate: int, fig_output_root: str):
    # cngt_clips = [file for file in os.listdir(cngt_root) if file.endswith('mpg') or file.endswith('.mov')]

    # clip_durations = []
    # for clip in tqdm(cngt_clips):
    #     start_ms = int(clip.split("_")[4])
    #     end_ms = int(clip.split("_")[5])
    #     n_frames = math.ceil(framerate * (end_ms - start_ms) / 1000)
    #     clip_durations.append(n_frames)

    cngt_metadata = [file.replace("mpg", "gzip") for file in os.listdir(cngt_root) if file.endswith('mpg') or file.endswith('.mov')]

    clip_durations = []
    for file in cngt_metadata:
        file_metadata = load_gzip(file)
        clip_durations.append(file_metadata["num_frames"])

    print_stats(clip_durations, framerate, "CNGT", fig_output_root)


def get_stats_signbank(signbank_root: str, framerate: int, fig_output_root: str):
    sb_clip_paths = [os.path.join(signbank_root, file) for file in os.listdir(signbank_root) if
                     file.endswith('mp4') or file.endswith('.mov')]

    clip_durations = []
    for clip in tqdm(sb_clip_paths):
        vcap = cv2.VideoCapture(clip)
        n_frames = 0
        while True:
            success, _ = vcap.read()
            if not success:
                break
            n_frames += 1

        clip_durations.append(n_frames)

    print_stats(clip_durations, framerate, "Signbank", fig_output_root)


def main(params):
    root = params.root
    cngt_folder = params.cngt_folder
    sb_folder = params.sb_folder
    framerate = params.framerate
    fig_output_root = params.fig_output_root

    cngt_root = os.path.join(root, cngt_folder)
    sb_root = os.path.join(root, sb_folder)

    assert os.path.exists(cngt_root), f"{cngt_root} doesn't exist, please make sure the given root is correct"
    assert os.path.exists(sb_root), f"{sb_root} doesn't exist, please make sure the given root is correct"

    get_stats_cngt(cngt_root, framerate, fig_output_root)
    # get_stats_signbank(sb_root, framerate, fig_output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="D:/Thesis/datasets"
    )

    parser.add_argument(
        "--cngt_folder",
        type=str,
        default="cngt_single_signs"
    )

    parser.add_argument(
        "--sb_folder",
        type=str,
        default="NGT_Signbank_resized"
    )

    parser.add_argument(
        "--fig_output_root",
        type=str,
        default="D:/Thesis/graphs"
    )

    parser.add_argument(
        "--framerate",
        type=int,
        default="25"
    )

    params, _ = parser.parse_known_args()
    main(params)
