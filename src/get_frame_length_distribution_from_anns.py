import os
from tqdm import tqdm
import argparse
import pympi
from multiprocessing import Pool
from pathlib import Path
from zipfile import ZipFile
import math
import shutil

from src.utils.util import save_gzip, count_video_frames, load_gzip
from intervaltree import Interval, IntervalTree

from src.utils.parse_cngt_glosses import parse_cngt_gloss

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main(params):

    # read user arguments
    root = params.root
    cngt_folder = params.cngt_folder
    sb_vocab_file = params.sb_vocab_file

    # build the entire paths for the datasets
    cngt_root = os.path.join(root, cngt_folder)
    sb_vocab_path = os.path.join(root, sb_vocab_file)

    sb_vocab = load_gzip(sb_vocab_path)

    print(f"Getting frame length distribution of CNGT from annotations in {cngt_root}")

    framerate = 25

    ann_filenames = [file for file in os.listdir(cngt_root) if file.endswith('.eaf')]

    ann_duration_frames = []
    for ann_file in tqdm(ann_filenames):
        ann_file = pympi.Elan.Eaf(os.path.join(cngt_root, ann_file))

        glosses_lefth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[0])
        glosses_righth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[1])

        right_intervalTree = IntervalTree()

        if glosses_righth:
            for ann in glosses_righth:
                start_ms, end_ms, gloss = ann[0], ann[1], ann[2]
                parsed_gloss = parse_cngt_gloss(gloss, sb_vocab)

                if parsed_gloss in sb_vocab['gloss_to_id'] and end_ms - start_ms != 0:
                    right_intervalTree.add(Interval(begin=start_ms, end=end_ms, data=parsed_gloss))

        merged_intervalTree = right_intervalTree.copy()

        if glosses_lefth:
            for ann in glosses_lefth:
                start_ms, stop_ms, gloss = ann[0], ann[1], ann[2]
                parsed_gloss = parse_cngt_gloss(gloss, sb_vocab)

                if parsed_gloss in sb_vocab['gloss_to_id'] and end_ms - start_ms != 0:
                    # for every annotation on the left hand, check for overlap with another annotation
                    overlaps = merged_intervalTree.overlap(start_ms, stop_ms)
                    if overlaps:
                        overlap_exceeded = False
                        for overlapping_interval in overlaps:

                            intersection = min(stop_ms, overlapping_interval.end) - max(start_ms, overlapping_interval.begin)
                            union = max(stop_ms, overlapping_interval.end) - min(start_ms, overlapping_interval.begin)
                            iou = intersection / union
                            # if the gloss overlaps a lot with one in the intervalTree, skip gloss
                            if iou >= 0.9:
                                overlap_exceeded = True
                                break
                        if overlap_exceeded:
                            continue

            if parsed_gloss in sb_vocab['gloss_to_id']:
                for interval in right_intervalTree:
                    if start_ms == interval.end and interval.data == parsed_gloss:
                        merged_intervalTree.remove(interval)
                        merged_intervalTree.add(Interval(interval.begin, stop_ms, parsed_gloss))

                    if stop_ms == interval.begin and interval.data == parsed_gloss:
                        merged_intervalTree.remove(interval)
                        merged_intervalTree.add(Interval(start_ms, interval.end, parsed_gloss))

        for interval_obj in merged_intervalTree:
            ann_duration_frames.append(math.ceil(framerate/1000*(interval_obj.end - interval_obj.begin)))

    print(len(ann_duration_frames))

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")
    colors = sns.color_palette('pastel')

    upper_quartile = np.percentile(ann_duration_frames, 75)
    lower_quartile = np.percentile(ann_duration_frames, 25)
    iqr = upper_quartile - lower_quartile

    ann_duration_frames = np.array(ann_duration_frames)

    upper_whisker = ann_duration_frames[np.where(ann_duration_frames <= upper_quartile + 1.5 * iqr, True, False)].max()
    lower_whisker = ann_duration_frames[np.where(ann_duration_frames >= lower_quartile - 1.5 * iqr, True, False)].min()

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)})

    x_ticks = np.arange(lower_whisker - 1, upper_whisker + 1)
    x_labels = [tick if tick % 2 != 0 else "" for tick in x_ticks]

    ax_box.boxplot(ann_duration_frames, showfliers=False, vert=False, showmeans=True)
    ax_box.set_yticks([])

    ax_hist.hist(ann_duration_frames, bins=np.arange(lower_whisker - 1, upper_whisker + 1)-0.5, align='mid', color=colors[0])

    plt.xlim([lower_whisker - 1, upper_whisker + 1])
    plt.gca().set_xticks(x_ticks)
    plt.gca().set_xticklabels(x_labels)
    plt.ylabel("Number of clips")
    plt.xlabel("Number of frames")
    plt.tight_layout()
    plt.show()


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
        default="CNGT_complete"
    )

    parser.add_argument(
        "--sb_vocab_file",
        type=str,
        default="signbank_vocab.gzip"
    )

    params, _ = parser.parse_known_args()
    main(params)
