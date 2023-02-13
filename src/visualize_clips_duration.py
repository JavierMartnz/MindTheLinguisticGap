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


def print_stats(frame_durations: list, framerate: int, dataset: str, fig_output_root: str):
    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    upper_quartile = np.percentile(frame_durations, 75)
    lower_quartile = np.percentile(frame_durations, 25)
    iqr = upper_quartile - lower_quartile

    frame_durations = np.array(frame_durations)

    upper_whisker = frame_durations[np.where(frame_durations <= upper_quartile + 1.5 * iqr, True, False)].max()
    lower_whisker = frame_durations[np.where(frame_durations >= lower_quartile - 1.5 * iqr, True, False)].min()

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)})

    ax_box.boxplot(frame_durations, showfliers=False, vert=False)
    ax_box.set_yticks([])

    ax_hist.hist(frame_durations, bins='auto', align='mid')
    plt.xlim([lower_whisker - 1, upper_whisker + 1])
    plt.gca().set_xticks(np.linspace(lower_whisker - 1, upper_whisker + 1, num=8, dtype=int))

    # plt.hist(frame_durations, bins='auto')
    # plt.yscale('log')
    # plt.axvline(mean, c='r', ls='--')
    # plt.axvspan(mean - std, mean + std, alpha=0.2, color='red')
    plt.suptitle(f"Number of frames per clip in {dataset} at {framerate} fps")
    plt.ylabel("Number of clips")
    plt.xlabel("Number of frames")
    plt.tight_layout()

    os.makedirs(fig_output_root, exist_ok=True)
    plt.savefig(os.path.join(fig_output_root, f"{dataset}_clips_duration_{framerate}fps.png"))


def get_stats_cngt(cngt_root: str, framerate: int, fig_output_root: str):
    cngt_clips = [file for file in os.listdir(cngt_root) if file.endswith('mpg') or file.endswith('.mov')]

    frame_durations = []
    for clip in tqdm(cngt_clips):
        start_ms = int(clip.split("_")[4])
        end_ms = int(clip.split("_")[5])
        gloss = clip.split("_")[-2]
        duration_frames = math.ceil(framerate * (end_ms - start_ms) / 1000)
        frame_durations.append(duration_frames)

    print_stats(frame_durations, framerate, "CNGT", fig_output_root)


def get_stats_signbank(signbank_root: str, framerate: int, fig_output_root: str):
    sb_clip_paths = [os.path.join(signbank_root, file) for file in os.listdir(signbank_root) if
                     file.endswith('mp4') or file.endswith('.mov')]

    frame_durations = []
    for clip in tqdm(sb_clip_paths):
        vcap = cv2.VideoCapture(clip)
        frames = 0
        while True:
            success, _ = vcap.read()
            if not success:
                break
            frames += 1

        frame_durations.append(frames)

    print_stats(frame_durations, framerate, "Signbank", fig_output_root)


def main(params):
    root = params.root
    cngt_folder = params.cngt_folder
    sb_folder = params.sb_folder
    framerate = params.framerate
    fig_output_root = params.fig_output_root

    cngt_root = os.path.join(root, cngt_folder)
    sb_root = os.path.join(root, sb_folder)

    get_stats_cngt(cngt_root, framerate, fig_output_root)
    get_stats_signbank(sb_root, framerate, fig_output_root)


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
