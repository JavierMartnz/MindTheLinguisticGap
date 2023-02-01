import os
import cv2
from utils.util import load_gzip, make_dir, save_gzip, count_video_frames
from utils.parse_cngt_glosses import parse_cngt_gloss
import pympi
from intervaltree import Interval, IntervalTree
import math
# import numpy as np
import argparse
from pathlib import Path
import shutil
from multiprocessing import Pool
from tqdm import tqdm
# import subprocess 
from zipfile import ZipFile
import random
from utils.util import extract_zip


def trim_clip(input_filename, start_time, end_time, start_frame, end_frame, gloss, cls, output_root, trim_format="%.3f"):

    start_time /= 1000
    end_time /= 1000

    # windows forbids filanames with semicolon, so we need to change how those files are stored
    gloss = gloss.replace(":", ";")

    filename = "%s_%s_%s_%s_%s_%s_%s.mpg" % (
        Path(input_filename).stem,
        trim_format % start_time,
        trim_format % end_time,
        start_frame,
        end_frame,
        gloss,
        cls
    )

    output_filename = os.path.join(output_root, filename)

    # if the video already exists, there's no point in processing the video
    if os.path.exists(output_filename):
        return None

    if os.path.exists(input_filename):
        if not os.path.isdir(output_root):
            make_dir(output_root)

        # Construct command to trim the videos (ffmpeg required).
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel panic",
            '-y',
            "-i",
            input_filename,
            "-ss",
            trim_format % start_time,
            "-t",
            trim_format % (end_time - start_time),
            "-preset ultrafast",
            '%s' % output_filename,
        ]

        command = " ".join(command)

        os.system(command)

        # Check if the video was successfully saved.
        status = os.path.exists(output_filename)
        if not status:
            print(output_filename + ' not downloaded')
            return None

    return output_filename


def process_file_for_trimming(file, dataset_root, signbank_vocab_path, output_root, window_size):
    if file.endswith('.mpg'):

        file_path = os.path.join(dataset_root, file)
        ann_path = file_path[:-3] + 'eaf'

        # check if the associated annotation file exists
        if not os.path.exists(ann_path):
            print(f"Early return: video {file} does not have an associated annotation file")
            return

        vcap = cv2.VideoCapture(file_path)

        num_video_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)

        if num_video_frames <= 0:
            print('Early return: video has zero frames')
            return

        signbank_vocab = load_gzip(signbank_vocab_path)
        sb_glosses = list(signbank_vocab['gloss_to_id'].keys())

        ann_file = pympi.Elan.Eaf(ann_path)

        glosses_lefth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[0])
        glosses_righth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[1])

        right_intervalTree = IntervalTree()

        if glosses_righth:

            for ann in glosses_righth:
                start_ms, stop_ms = ann[0], ann[1]
                gloss = ann[2]

                parsed_gloss = parse_cngt_gloss(gloss, sb_glosses)

                start_frame = math.ceil(25.0 * (start_ms / 1000.0))
                stop_frame = math.floor(25.0 * (stop_ms / 1000.0)) + 1

                if start_frame > num_video_frames or stop_frame > num_video_frames:
                    return

                data = {'parsed_gloss': parsed_gloss, 'start_ms': start_ms, 'stop_ms': stop_ms}

                interval = Interval(begin=start_ms, end=stop_ms, data=data)

                if parsed_gloss in signbank_vocab['gloss_to_id']:
                    right_intervalTree.add(interval)

        merged_intervalTree = right_intervalTree.copy()

        if glosses_lefth:

            for ann in glosses_lefth:
                start_ms, stop_ms = ann[0], ann[1]
                gloss = ann[2]
                parsed_gloss = parse_cngt_gloss(gloss, sb_glosses)
                start_frame = math.ceil(25.0 * (start_ms / 1000.0))
                stop_frame = math.floor(25.0 * (stop_ms / 1000.0)) + 1

                if start_frame > num_video_frames or stop_frame > num_video_frames:
                    return

                begin = start_frame
                end = stop_frame

                # these code section stops duplicated annotation of glosses in different hands
                overlaps = right_intervalTree.overlap(begin, end)
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
                        return

                if parsed_gloss in signbank_vocab['gloss_to_id']:
                    for interval in right_intervalTree:

                        data = {'parsed_gloss': parsed_gloss}

                        if begin == interval.end and interval.data['parsed_gloss'] == parsed_gloss:
                            data['start_ms'] = interval.data['start_ms']
                            data['stop_ms'] = stop_ms
                            merged_interval = Interval(interval.begin, end, data)
                            merged_intervalTree.remove(interval)
                            merged_intervalTree.add(merged_interval)

                        if end == interval.begin and interval.data == parsed_gloss:
                            data['start_ms'] = start_ms
                            data['stop_ms'] = interval.data['stop_ms']
                            merged_interval = Interval(begin, interval.end, data)
                            merged_intervalTree.remove(interval)
                            merged_intervalTree.add(merged_interval)

        for interval_obj in merged_intervalTree:
            trimmed_filename = trim_clip(file_path, interval_obj.data['start_ms'], interval_obj.data['stop_ms'],
                                         interval_obj.begin, interval_obj.end, interval_obj.data['parsed_gloss'],
                                         signbank_vocab['gloss_to_id'][interval_obj.data['parsed_gloss']], output_root)

            if not trimmed_filename is None:
                # since opencv's number of frames is unreliable, we count the frames ourselves
                num_trimmed_frames = count_video_frames(trimmed_filename)
                # we create metadata that will be helpful for the loading
                metadata = {"num_frames": num_trimmed_frames, "start_frames": []}
                num_clips = math.ceil(num_trimmed_frames / window_size)
                for j in range(num_clips):
                    metadata["start_frames"].append(j * window_size)

                save_gzip(metadata, trimmed_filename[:-3] + 'gzip')

def main(params):
    root = params.root()
    dataset_root = params.dataset_root
    signbank_vocab_file = params.signbank_vocab_path
    output_root = params.output_root
    window_size = params.window_size

    dataset_path = os.path.join(root, dataset_root)
    signbank_vocab_path = os.path.join(root, signbank_vocab_file)
    output_path = os.path.join(root, output_root)

    dataset_zip = dataset_path + ".zip"

    if os.path.isfile(dataset_zip):
        extract_zip(dataset_zip)

    make_dir(output_path)

    all_files = os.listdir(dataset_path)

    # multiprocessing bit based on https://github.com/tqdm/tqdm/issues/484
    pool = Pool()
    pbar = tqdm(total=len(all_files))

    def update(*a):
        pbar.update()

    for i in range(pbar.total):
        pool.apply_async(process_file_for_trimming,
                         args=(all_files[i], dataset_path, signbank_vocab_path, output_path, int(window_size)),
                         callback=update)

    pool.close()
    pool.join()

    # zip the resulting folder
    print(f"Zipping the files in {output_path}")
    zip_basedir = Path(output_path).parent
    zip_filename = os.path.basename(output_path) + '.zip'

    with ZipFile(os.path.join(zip_basedir, zip_filename), 'w') as zipfile:
        for filename in tqdm(os.listdir(output_path), position=0, leave=True):
            zipfile.write(os.path.join(output_path, filename), filename)

    # just delete the previous directory is the zip file was created
    if os.path.isfile(os.path.join(zip_basedir, zip_filename)):
        print("Zipfile was successfully created")

    # once the files are in output folder, we are going to create the splits
    # datapoints = [file[:-4] for file in os.listdir(output_root) if file.endswith('.mpg')]
    #
    # random.seed(42)
    # random.shuffle(datapoints)
    #
    # train_val_idx = int(len(datapoints) * (4 / 6))
    # val_test_idx = int(len(datapoints) * (5 / 6))
    #
    # data = {"train": datapoints[:train_val_idx], "val": datapoints[train_val_idx:val_test_idx], "test": datapoints[val_test_idx:]}
    #
    # print(f"Split sizes:\n\t-train={len(data['train'])}\n\t-val={len(data['val'])}\n\t-test={len(data['test'])}")
    #
    # splits = ["train", "val", "test"]
    #
    # for split in splits:
    #     print(f"Creating {split} split...")
    #
    #     # create the folder for the split
    #     video_split_root = os.path.join(output_root, split)
    #     os.makedirs(video_split_root, exist_ok=True)
    #
    #     # iterate over all datapoints and move the videos and annotation files to the new folder
    #     for datapoint in tqdm(data[split]):
    #         video = datapoint + ".mpg"
    #         metadata = datapoint + ".gzip"
    #         shutil.move(os.path.join(output_root, video), os.path.join(video_split_root, video))
    #         shutil.move(os.path.join(output_root, metadata), os.path.join(video_split_root, metadata))
    #
    # # we zip the resulting folder and delete the original one since Ponyland backups break with smaller files
    # # print("Start zipping isolated clips...")
    # # zip_basedir = Path(output_root).parent
    # # zip_filename = os.path.basename(output_root) + '.zip'
    # #
    # # all_filenames = os.listdir(output_root)
    # #
    # # # training
    # # with ZipFile(os.path.join(zip_basedir, zip_filename), 'w') as zipfile:
    # #     for filename in tqdm(all_filenames):
    # #         zipfile.write(os.path.join(output_root, filename), filename)
    # #
    # #         # just delete the previous directory is the zip file was created
    # # if os.path.isfile(os.path.join(zip_basedir, zip_filename)):
    # #     print("Zipfile was successfully created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="D:/Thesis/datasets"
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="CNGT_final_512res"
    )

    parser.add_argument(
        "--signbank_vocab_file",
        type=str,
        default="signbank_vocab.gzip"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="cngt_single_signs_512"
    )

    parser.add_argument(
        "--window_size",
        type=str,
        default="16"
    )

    params, _ = parser.parse_known_args()
    main(params)
