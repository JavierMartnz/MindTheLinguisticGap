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

def get_signers_dict(ann_object):
    signers = {}

    for tier in ann_object.tiers:

        if tier[-2:] == "S1" and "PARTICIPANT" in ann_object.tiers[tier][2].keys():
            if "S1" not in signers:
                signers["S1"] = ann_object.tiers[tier][2]["PARTICIPANT"]
        elif tier[-2:] == "S2" and "PARTICIPANT" in ann_object.tiers[tier][2].keys():
            if "S2" not in signers:
                signers["S2"] = ann_object.tiers[tier][2]["PARTICIPANT"]

    return signers

def count_glosses(glosses_righth, glosses_lefth, sb_vocab):

    num_glosses = 0

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
        num_glosses += 1

    return num_glosses

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

    signers_count_dict = {}
    ann_duration_frames = []
    for ann_file in tqdm(ann_filenames):
        ann_file = pympi.Elan.Eaf(os.path.join(cngt_root, ann_file))

        ann_signers_dict = get_signers_dict(ann_file)

        # the tiers containing glosses are ordered as 'GlossL S1', 'GlossL S2', 'GlossR S1', 'GlossR S2'
        s1_glosses_lefth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[0])
        s1_glosses_righth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[2])
        s2_glosses_lefth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[1])
        s2_glosses_righth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[3])

        s1_num_glosses = count_glosses(s1_glosses_righth, s1_glosses_lefth, sb_vocab)
        s2_num_glosses = count_glosses(s2_glosses_righth, s2_glosses_lefth, sb_vocab)

        # sometimes there are no annotations for both signers, hence this if clause
        if 'S1' in ann_signers_dict.keys():
            signers_count_dict[ann_signers_dict['S1']] = signers_count_dict.get(ann_signers_dict['S1'], 0) + s1_num_glosses
        if 'S2' in ann_signers_dict.keys():
            signers_count_dict[ann_signers_dict['S2']] = signers_count_dict.get(ann_signers_dict['S2'], 0) + s2_num_glosses

    sorted_signers_count_dict = {k: v for k, v in sorted(signers_count_dict.items(), key=lambda item: item[0])}

    print(f"The CNGT has a total of {len(sorted_signers_count_dict.keys())} signers.")

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")
    colors = sns.color_palette('pastel')

    plt.bar(np.arange(len(sorted_signers_count_dict.values())), sorted_signers_count_dict.values(), color=colors[0])
    plt.ylabel("Number of glosses")
    plt.xlabel("Signer")
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
