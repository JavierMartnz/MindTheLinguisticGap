import pympi
import os
from tqdm import tqdm
import cv2
import numpy as np
from intervaltree import IntervalTree, Interval
import math

from parse_cngt_glosses import parse_cngt_gloss
from util import load_gzip, save_gzip


def get_gloss_occurrences(glosses_righth, glosses_lefth, signbank_vocab, num_video_frames, cngt=True):
    glosses = ["" for i in range(num_video_frames)]

    left_intervalTree = IntervalTree()
    right_intervalTree = IntervalTree()

    if glosses_righth:

        for ann in glosses_righth:
            start_ms, stop_ms = ann[0], ann[1]

            if start_ms == stop_ms:
                continue

            gloss = ann[2]
            if cngt:
                parsed_gloss = parse_cngt_gloss(gloss, signbank_vocab['glosses'])
            else:
                parsed_gloss = gloss
            start_frame = math.ceil(25.0 * (start_ms / 1000.0))
            stop_frame = math.floor(25.0 * (stop_ms / 1000.0)) + 1

            if start_frame > num_video_frames or stop_frame > num_video_frames:
                continue

            interval = Interval(begin=start_ms, end=stop_ms, data=parsed_gloss)

            if cngt:
                if parsed_gloss in signbank_vocab['gloss_to_id']:
                    right_intervalTree.add(interval)
                    for i in range(start_frame, stop_frame):
                        glosses[i] = parsed_gloss
            else:
                right_intervalTree.add(interval)
                for i in range(start_frame, stop_frame):
                    glosses[i] = parsed_gloss

    if glosses_lefth:

        for ann in glosses_lefth:
            start_ms, stop_ms = ann[0], ann[1]
            gloss = ann[2]
            if cngt:
                parsed_gloss = parse_cngt_gloss(gloss, signbank_vocab['glosses'])
            else:
                parsed_gloss = gloss
            start_frame = math.ceil(25.0 * (start_ms / 1000.0))
            stop_frame = math.floor(25.0 * (stop_ms / 1000.0)) + 1

            if start_frame > num_video_frames or stop_frame > num_video_frames:
                continue

            begin = start_ms
            end = stop_ms

            # these code section stops duplicated annotation of glosses in different hands
            overlaps = left_intervalTree.overlap(begin, end)
            if overlaps:
                overlap_exceeded = False
                for interval in overlaps:
                    intersection = min(end, interval.end) - max(begin, interval.begin)
                    union = max(end, interval.end) - min(begin, interval.begin)
                    iou = intersection / union
                    # if the gloss overlaps a lot with one in the intervalTree, skip gloss
                    if iou >= 0.9:
                        overlap_exceeded = True
                        break
                if overlap_exceeded:
                    continue
            if cngt:
                if parsed_gloss in signbank_vocab['gloss_to_id']:
                    for i in range(start_frame, stop_frame):
                        glosses[i] = parsed_gloss
            else:
                for i in range(start_frame, stop_frame):
                    glosses[i] = parsed_gloss

    past_frame = ''
    gloss_occurrences = []

    for frame in glosses:
        current_frame = frame

        if past_frame != current_frame and current_frame != '':
            gloss_occurrences.append(current_frame)

        past_frame = current_frame

    return gloss_occurrences


def main():
    cngt_root = "D:/Thesis/datasets/CNGT_split"
    signbank_vocab_path = "D:/Thesis/datasets/signbank_vocab.gzip"
    video_files = [file for file in os.listdir(cngt_root) if file.endswith(".mpg")]

    signer_dict = {}

    for file in tqdm(video_files):

        signer = file.split('_')[0]

        file_path = os.path.join(cngt_root, file)
        ann_path = file_path[:-3] + 'eaf'

        # check if the associated annotation file exists
        if not os.path.exists(ann_path):
            print(f"Early return: video {file} does not have an associated annotation file")
            continue

        vcap = cv2.VideoCapture(file_path)
        num_video_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_video_frames <= 0:
            print('Early return: video has zero frames')
            continue

        signbank_vocab = load_gzip(signbank_vocab_path)

        ann_file = pympi.Elan.Eaf(ann_path)

        glosses_lefth1 = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[0])
        glosses_righth1 = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[1])

        occ = get_gloss_occurrences(glosses_lefth1, glosses_righth1, signbank_vocab, num_video_frames, cngt=False)

        if occ is not None:
            if signer not in signer_dict.keys():
                signer_dict[signer] = occ
            else:
                signer_dict[signer].extend(occ)

    save_gzip(signer_dict, "D:/Thesis/anns_per_signer.gzip")
    print(f"Dictionary of glosses per signer saved successfully.")


if __name__ == "__main__":
    main()
