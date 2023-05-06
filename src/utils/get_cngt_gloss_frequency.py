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


def get_gloss_freq(glosses_righth, glosses_lefth, sb_vocab, gloss_count_dict, naive=False):

    if naive:

        if glosses_righth:
            for ann in glosses_righth:
                start_ms, end_ms, gloss = ann[0], ann[1], ann[2]
                # parsed_gloss = parse_cngt_gloss(gloss, sb_vocab)

                if end_ms - start_ms != 0:
                    gloss_count_dict[gloss] = gloss_count_dict.get(gloss, 0) + 1

        if glosses_lefth:
            for ann in glosses_lefth:
                start_ms, end_ms, gloss = ann[0], ann[1], ann[2]
                # parsed_gloss = parse_cngt_gloss(gloss, sb_vocab)

                if end_ms - start_ms != 0:
                    gloss_count_dict[gloss] = gloss_count_dict.get(gloss, 0) + 1

    else:
        right_intervalTree = IntervalTree()
        if glosses_righth:
            for ann in glosses_righth:
                start_ms, end_ms, gloss = ann[0], ann[1], ann[2]
                # parsed_gloss = parse_cngt_gloss(gloss, sb_vocab)

                if gloss in sb_vocab['gloss_to_id'] and end_ms - start_ms != 0:
                    right_intervalTree.add(Interval(begin=start_ms, end=end_ms, data=gloss))

        merged_intervalTree = right_intervalTree.copy()

        if glosses_lefth:
            for ann in glosses_lefth:
                start_ms, stop_ms, gloss = ann[0], ann[1], ann[2]
                # parsed_gloss = parse_cngt_gloss(gloss, sb_vocab)

                if gloss in sb_vocab['gloss_to_id'] and stop_ms - start_ms != 0:
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

            if gloss in sb_vocab['gloss_to_id']:
                for interval in right_intervalTree:
                    if start_ms == interval.end and interval.data == gloss:
                        merged_intervalTree.remove(interval)
                        merged_intervalTree.add(Interval(interval.begin, stop_ms, gloss))

                    if stop_ms == interval.begin and interval.data == gloss:
                        merged_intervalTree.remove(interval)
                        merged_intervalTree.add(Interval(start_ms, interval.end, gloss))

        for interval_obj in merged_intervalTree:
            gloss_count_dict[interval_obj.data] = gloss_count_dict.get(interval_obj.data, 0) + 1

    return gloss_count_dict


def main(params):
    # read user arguments
    root = params.root
    cngt_folder = params.cngt_folder
    sb_vocab_file = params.sb_vocab_file

    # build the entire paths for the datasets
    cngt_root = os.path.join(root, cngt_folder)
    sb_vocab_path = os.path.join(root, sb_vocab_file)

    sb_vocab = load_gzip(sb_vocab_path)

    print(f"Getting gloss frequency of CNGT from annotations in {cngt_root}")

    framerate = 25

    ann_filenames = [file for file in os.listdir(cngt_root) if file.endswith('.eaf')]

    gloss_count_dict = {}
    for ann_file in tqdm(ann_filenames):
        ann_file = pympi.Elan.Eaf(os.path.join(cngt_root, ann_file))

        # the tiers containing glosses are ordered as 'GlossL S1', 'GlossL S2', 'GlossR S1', 'GlossR S2'
        s1_glosses_lefth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[0])
        s1_glosses_righth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[2])
        s2_glosses_lefth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[1])
        s2_glosses_righth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[3])

        # update the dictionary with glosses annotated for both signers
        gloss_count_dict = get_gloss_freq(s1_glosses_righth, s1_glosses_lefth, sb_vocab, gloss_count_dict, naive=True)
        gloss_count_dict = get_gloss_freq(s2_glosses_righth, s2_glosses_lefth, sb_vocab, gloss_count_dict, naive=True)

    sorted_gloss_count_dict = {k: v for k, v in sorted(gloss_count_dict.items(), key=lambda item: item[1], reverse=True)}
    sorted_gloss_count_dict = {k: v for k, v in sorted_gloss_count_dict.items() if v > 100}

    print(f"CNGT has a total of {len(set(gloss_count_dict.keys()))} unique glosses.")
    print(f"CNGT has a total of {sum(list(gloss_count_dict.values()))} annotations, with an average of {np.mean(list(gloss_count_dict.values()))} annotations per sign")

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")
    colors = sns.color_palette('pastel')

    plt.bar(np.arange(len(sorted_gloss_count_dict.values())), sorted_gloss_count_dict.values(), color=colors[0])
    plt.ylabel("Frequency")
    plt.xlabel("Gloss")
    plt.xticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="C:/Thesis/datasets"
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
