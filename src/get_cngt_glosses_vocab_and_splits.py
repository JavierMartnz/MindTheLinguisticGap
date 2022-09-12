import os
import random
import shutil
import cv2
from utils.util import load_gzip, save_gzip, make_dir, save_vocab
from utils.parse_cngt_glosses import parse_cngt_gloss
import pympi
from intervaltree import Interval, IntervalTree
import math
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile


def main(params):
    dataset_root = params.dataset_root
    signbank_vocab_path = params.signbank_vocab_path
    gloss_output_root = params.gloss_output_root
    cngt_vocab_path = params.cngt_vocab_path
    log_filename = params.log_filename
    cngt_output_root = params.cngt_output_root

    cngt_gloss_to_id = {}
    no_gloss_videos = 0
    num_videos = 0
    log_summary = ""

    all_files = os.listdir(dataset_root)

    print("Processing glosses...")

    for file in tqdm(all_files):

        if file.endswith('.mpg'):

            num_videos += 1
            file_path = os.path.join(dataset_root, file)
            ann_path = file_path[:-3] + 'eaf'

            # check if the associated annotation file exists
            if not os.path.exists(ann_path):
                print(f"Early return: video {file} does not have an associated annotation file")
                continue

            # check that the video file is not empty
            vcap = cv2.VideoCapture(file_path)
            num_video_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            if num_video_frames <= 0:
                print('Early return: video has zero frames')
                continue

            signbank_vocab = load_gzip(signbank_vocab_path)

            ann_file = pympi.Elan.Eaf(ann_path)

            # we're just working with glosses, so get the annotations for both hands separately
            glosses_lefth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[0])
            glosses_righth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[1])

            cls = np.full(num_video_frames, -1, dtype=int)
            glosses = ["" for i in range(num_video_frames)]

            # these interval tree will help us identify overlap in annotations and avoid counting them twice
            left_intervalTree = IntervalTree()
            right_intervalTree = IntervalTree()

            if glosses_righth:

                for ann in glosses_righth:
                    start_ms, stop_ms = ann[0], ann[1]
                    gloss = ann[2]
                    # the function below will get rid of glosses that are not in the given vocabulary and that do not satisfy a set of rules
                    parsed_gloss = parse_cngt_gloss(gloss, signbank_vocab['glosses'])
                    start_frame = math.ceil(25.0 * (start_ms / 1000.0))
                    stop_frame = math.floor(25.0 * (stop_ms / 1000.0)) + 1

                    if start_frame > num_video_frames or stop_frame > num_video_frames:
                        continue

                    interval = Interval(begin=start_ms, end=stop_ms, data=parsed_gloss)

                    if parsed_gloss in signbank_vocab['gloss_to_id']:
                        right_intervalTree.add(interval)
                        cls[start_frame:stop_frame] = signbank_vocab['gloss_to_id'][parsed_gloss]
                        # store for vocab
                        cngt_gloss_to_id[parsed_gloss] = signbank_vocab['gloss_to_id'][parsed_gloss]
                        for i in range(start_frame, stop_frame):
                            glosses[i] = parsed_gloss

            if glosses_lefth:

                for ann in glosses_lefth:
                    start_ms, stop_ms = ann[0], ann[1]
                    gloss = ann[2]
                    parsed_gloss = parse_cngt_gloss(gloss, signbank_vocab['glosses'])
                    start_frame = math.ceil(25.0 * (start_ms / 1000.0))
                    stop_frame = math.floor(25.0 * (stop_ms / 1000.0)) + 1

                    if start_frame > num_video_frames or stop_frame > num_video_frames:
                        continue

                    begin = start_ms
                    end = stop_ms

                    # these code section avoids duplicated annotation of glosses in different hands
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

                    if parsed_gloss in signbank_vocab['gloss_to_id']:
                        cls[start_frame:stop_frame] = signbank_vocab['gloss_to_id'][parsed_gloss]
                        # store for vocab
                        cngt_gloss_to_id[parsed_gloss] = signbank_vocab['gloss_to_id'][parsed_gloss]
                        for i in range(start_frame, stop_frame):
                            glosses[i] = parsed_gloss

            if all(cls == -1):
                no_gloss_videos += 1
                log_summary += f"{os.path.basename(file_path)}\n"
                continue

            save_data = {'cls': cls, 'glosses': glosses}

            gloss_path = os.path.join(gloss_output_root, file.replace('.mpg', '.gzip'))
            make_dir(gloss_output_root)  # make directory in case it doesn't exist
            save_gzip(save_data, gloss_path)

    logfile_path = os.path.join(gloss_output_root, log_filename)
    print(f"Saving summary file to {logfile_path}")
    if os.path.exists(logfile_path):
        os.remove(logfile_path)
    with open(logfile_path, 'w') as f:
        print(f"{no_gloss_videos} out of {num_videos} videos discarded because they contained no gloss annotations:\n\n" + log_summary, file=f)
    print(f"Saving CNGT vocabulary file to {cngt_vocab_path}")
    # after iterating over every file, save the vocab
    save_vocab(cngt_gloss_to_id, cngt_vocab_path)

    # now create the train/val/test splits
    # at this point of the code we need to split into test val test in a 4:1:1 ratio
    print("Starting train/val/test split...")

    datapoints = [file[:-5] for file in os.listdir(gloss_output_root) if file.endswith(".gzip")]

    random.seed(42)
    random.shuffle(datapoints)

    train_val_idx = int(len(datapoints) * (4 / 6))
    val_test_idx = int(len(datapoints) * (5 / 6))

    train_data = datapoints[:train_val_idx]
    val_data = datapoints[train_val_idx:val_test_idx]
    test_data = datapoints[val_test_idx:]

    print(f"Split sizes:\n\t-train={len(train_data)}\n\t-val={len(val_data)}\n\t-test={len(test_data)}")

    gloss_train_root = os.path.join(gloss_output_root, "train")
    gloss_val_root = os.path.join(gloss_output_root, "val")
    gloss_test_root = os.path.join(gloss_output_root, "test")
    video_train_root = os.path.join(cngt_output_root, "train")
    video_val_root = os.path.join(cngt_output_root, "val")
    video_test_root = os.path.join(cngt_output_root, "test")

    os.makedirs(gloss_train_root, exist_ok=True)
    os.makedirs(gloss_val_root, exist_ok=True)
    os.makedirs(gloss_test_root, exist_ok=True)
    os.makedirs(video_train_root, exist_ok=True)
    os.makedirs(video_val_root, exist_ok=True)
    os.makedirs(video_test_root, exist_ok=True)

    print("Creating training splits...")
    for datapoint in tqdm(train_data):
        video = datapoint + ".mpg"
        ann = datapoint + ".eaf"
        gloss = datapoint + ".gzip"
        shutil.copy(os.path.join(dataset_root, video), os.path.join(video_train_root, video))
        shutil.copy(os.path.join(dataset_root, ann), os.path.join(video_train_root, ann))
        shutil.move(os.path.join(gloss_output_root, gloss), os.path.join(gloss_train_root, gloss))

    print("Creating validation splits...")
    for datapoint in tqdm(val_data):
        video = datapoint + ".mpg"
        ann = datapoint + ".eaf"
        gloss = datapoint + ".gzip"
        shutil.copy(os.path.join(dataset_root, video), os.path.join(video_val_root, video))
        shutil.copy(os.path.join(dataset_root, ann), os.path.join(video_val_root, ann))
        shutil.move(os.path.join(gloss_output_root, gloss), os.path.join(gloss_val_root, gloss))

    print("Creating test splits...")
    for datapoint in tqdm(test_data):
        video = datapoint + ".mpg"
        ann = datapoint + ".eaf"
        gloss = datapoint + ".gzip"
        shutil.copy(os.path.join(dataset_root, video), os.path.join(video_test_root, video))
        shutil.copy(os.path.join(dataset_root, ann), os.path.join(video_test_root, ann))
        shutil.move(os.path.join(gloss_output_root, gloss), os.path.join(gloss_test_root, gloss))

    print("Zipping SignBank resized videos")

    zip_basedir = Path(gloss_output_root).parent
    zip_filename = os.path.basename(gloss_output_root) + '.zip'

    with ZipFile(os.path.join(zip_basedir, zip_filename), 'w') as zipfile:
        for subdir, _, filenames in os.walk(gloss_output_root):
            for filename in tqdm(filenames):
                if filename.endswith('gzip'):
                    zipfile.write(os.path.join(subdir, filename), os.path.join(os.path.basename(subdir), filename))

    if os.path.isfile(os.path.join(zip_basedir, zip_filename)):
        # maybe remove in a future
        print("Zipfile with glosses saved succesfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="D:/Thesis/datasets/CNGT_isolated_signers"
    )

    parser.add_argument(
        "--signbank_vocab_path",
        type=str,
        default="D:/Thesis/datasets/signbank_vocab.gzip"
    )

    parser.add_argument(
        "--gloss_output_root",
        type=str,
        default="D:/Thesis/datasets/glosses"
    )

    parser.add_argument(
        "--cngt_vocab_path",
        type=str,
        default="D:/Thesis/datasets/cngt_vocab.gzip"
    )

    parser.add_argument(
        "--cngt_output_root",
        type=str,
        default="D:/Thesis/datasets/CNGT_final"
    )

    parser.add_argument(
        "--log_filename",
        type=str,
        default="gloss_extraction_summary.txt"
    )

    params, _ = parser.parse_known_args()
    main(params)
