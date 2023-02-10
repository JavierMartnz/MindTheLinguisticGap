import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from tqdm import tqdm


def get_stats_cngt(cngt_root, fig: int):
    cngt_clips = [file for file in os.listdir(cngt_root) if file.endswith('mpg')]

    frame_durations = []
    filtered_duration_frames = []
    for clip in tqdm(cngt_clips):
        # clip_path = os.path.join(cngt_root, clip)
        # vcap = cv2.VideoCapture(clip_path)
        # frames = 0
        # while True:
        #     success, _ = vcap.read()
        #     if not success:
        #         break
        #     frames += 1

        start_ms = int(clip.split("_")[4])
        end_ms = int(clip.split("_")[5])
        gloss = clip.split("_")[-2]
        duration_frames = math.ceil(25 * (end_ms - start_ms) / 1000)
        if duration_frames > 1:
            filtered_duration_frames.append(duration_frames)
        frame_durations.append(duration_frames)

    n, bins = np.histogram(filtered_duration_frames)
    mids = 0.5 * (bins[1:] + bins[:-1])  # mid values of the bins
    mean = np.average(mids, weights=n)
    std = np.sqrt(np.average((mids - mean) ** 2, weights=n))

    print(f"The cngt clips have mean length of {round(mean, 1)} frames with std {round(std, 1)}")

    zoomed_duration_frames = np.array(filtered_duration_frames)
    zoomed_duration_frames = zoomed_duration_frames[np.where(zoomed_duration_frames < 50, True, False)]

    plt.figure(fig)
    plt.hist(zoomed_duration_frames, bins='auto')
    plt.axvline(mean, c='r', ls='--')
    plt.axvspan(mean - std, mean + std, alpha=0.2, color='red')
    plt.title("Histogram of number of frames in CNGT clips")
    plt.ylabel("Number of clips")
    plt.xlabel("Number of frames")


def get_stats_signbank(signbank_root, fig: int):
    clips = [file for file in os.listdir(signbank_root) if file.endswith('mp4')]

    frame_durations = []
    for clip in tqdm(clips):
        clip_path = os.path.join(signbank_root, clip)
        vcap = cv2.VideoCapture(clip_path)
        frames = 0
        while True:
            success, _ = vcap.read()
            if not success:
                break
            frames += 1

        frame_durations.append(frames)

    n, bins = np.histogram(frame_durations)
    mids = 0.5 * (bins[1:] + bins[:-1])  # mid values of the bins
    mean = np.average(mids, weights=n)
    std = np.sqrt(np.average((mids - mean) ** 2, weights=n))

    print(f"The Signbank clips have mean length of {round(mean, 1)} frames with std {round(std, 1)}")

    plt.figure(fig)
    plt.hist(frame_durations, bins='auto')
    plt.axvline(mean, c='r', ls='--')
    plt.axvspan(mean - std, mean + std, alpha=0.2, color='red')
    plt.title("Histogram of number of frames in Signbank clips")
    plt.ylabel("Number of clips")
    plt.xlabel("Number of frames")


def main():
    cngt_root = "D:/Thesis/datasets/cngt_single_signs"
    signbank_root = "D:/Thesis/datasets/NGT_Signbank_resized"

    get_stats_cngt(cngt_root, 0)
    get_stats_signbank(signbank_root, 1)
    plt.show()


if __name__ == "__main__":
    main()
