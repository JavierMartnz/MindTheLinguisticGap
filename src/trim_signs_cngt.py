import os
import cv2
from utils.util import load_gzip, save_gzip, count_video_frames
from utils.parse_cngt_glosses import parse_cngt_gloss
import pympi
from intervaltree import Interval, IntervalTree
import math
import numpy as np
import argparse
from pathlib import Path
import shutil
from multiprocessing import Pool
from tqdm import tqdm
# import subprocess 
from zipfile import ZipFile
import random
from utils.util import extract_zip


def trim_clip(input_filename, start_time, end_time, start_frame, end_frame, gloss, id, output_root, trim_format="%.3f"):

    start_time /= 1000
    end_time /= 1000

    # windows forbids filenames with semicolon, so we need to change how those files are stored, comment if using linux
    # gloss = gloss.replace(":", ";")

    filename = "%s_%s_%s_%s_%s_%s_%s.mpg" % (
        Path(input_filename).stem,
        trim_format % start_time,
        trim_format % end_time,
        start_frame,
        end_frame,
        gloss,
        id
    )

    output_filename = os.path.join(output_root, filename)

    # if the video already exists, there's no point in processing the video
    if os.path.exists(output_filename):
        return None

    if os.path.exists(input_filename):
        os.makedirs(output_root, exist_ok=True)

        # Construct command to trim the videos (ffmpeg required).
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel panic",
            '-y',
            "-i",
            input_filename,
            "-ss",
            str(start_time),
            "-t",
            str(end_time - start_time),
            output_filename,
        ]

        command = " ".join(command)
        os.system(command)

        # Check if the video was successfully saved.
        status = os.path.exists(output_filename)
        if not status:
            print(output_filename + ' not downloaded')
            return None

    return output_filename


def process_file_for_trimming(file, cngt_root, cngt_output_root, signbank_vocab_path, framerate, window_size):
    file_path = os.path.join(cngt_root, file)
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

    sb_vocab = load_gzip(signbank_vocab_path)
    ann_file = pympi.Elan.Eaf(ann_path)

    glosses_lefth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[0])
    glosses_righth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[1])

    right_intervalTree = IntervalTree()

    if glosses_righth:

        for ann in glosses_righth:
            start_ms, stop_ms = ann[0], ann[1]
            gloss = ann[2]

            # this line unifies glosses to Dutch
            parsed_gloss = parse_cngt_gloss(gloss, sb_vocab)

            start_frame = math.ceil(framerate * (start_ms / 1000.0))
            stop_frame = math.floor(framerate * (stop_ms / 1000.0)) + 1

            if start_frame > num_video_frames or stop_frame > num_video_frames:
                return

            data = {'parsed_gloss': parsed_gloss, 'start_ms': start_ms, 'stop_ms': stop_ms}

            interval = Interval(begin=start_ms, end=stop_ms, data=data)

            if parsed_gloss in sb_vocab['gloss_to_id']:
                right_intervalTree.add(interval)

    merged_intervalTree = right_intervalTree.copy()

    if glosses_lefth:

        for ann in glosses_lefth:
            start_ms, stop_ms = ann[0], ann[1]
            gloss = ann[2]
            parsed_gloss = parse_cngt_gloss(gloss, sb_vocab)
            start_frame = math.ceil(framerate * (start_ms / 1000.0))
            stop_frame = math.floor(framerate * (stop_ms / 1000.0)) + 1

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

            if parsed_gloss in sb_vocab['gloss_to_id']:
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
        trimmed_filename = trim_clip(file_path,
                                     interval_obj.data['start_ms'],
                                     interval_obj.data['stop_ms'],
                                     interval_obj.begin,
                                     interval_obj.end,
                                     interval_obj.data['parsed_gloss'],
                                     sb_vocab['gloss_to_id'][interval_obj.data['parsed_gloss']],
                                     cngt_output_root)

        # if a trimmed file was created, save the corresponding metadata
        if trimmed_filename is not None:
            # since opencv's number of frames is unreliable, we count the frames ourselves
            num_trimmed_frames = count_video_frames(trimmed_filename)
            # we create metadata that will be helpful for the loading
            metadata = {"num_frames": num_trimmed_frames, "start_frames": []}
            num_clips = math.ceil(num_trimmed_frames / window_size)
            for j in range(num_clips):
                metadata["start_frames"].append(j * window_size)

            save_gzip(metadata, trimmed_filename[:-3] + 'gzip')

def main(params):
    root = params.root
    cngt_folder = params.cngt_folder
    cngt_output_folder = params.cngt_output_folder
    signbank_vocab_file = params.signbank_vocab_file
    window_size = params.window_size
    framerate = params.framerate

    cngt_root = os.path.join(root, cngt_folder)
    cngt_output_root = os.path.join(root, cngt_output_folder)
    signbank_vocab_path = os.path.join(root, signbank_vocab_file)

    os.makedirs(cngt_output_root, exist_ok=True)
    all_videos = [file for file in os.listdir(cngt_root) if file.endswith(".mpg") or file.endswith(".mov")]

    print(f"Trimming clips in {cngt_root}\nand saving them in\n{cngt_output_root}")

    # multiprocessing bit based on https://github.com/tqdm/tqdm/issues/484
    pool = Pool()
    pbar = tqdm(total=len(all_videos))

    def update(*a):
        pbar.update()

    for i in range(pbar.total):
        pool.apply_async(process_file_for_trimming,
                         args=(all_videos[i],
                               cngt_root,
                               cngt_output_root,
                               signbank_vocab_path,
                               framerate,
                               window_size),
                         callback=update)

    pool.close()
    pool.join()


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
        default="CNGT_512"
    )

    parser.add_argument(
        "--cngt_output_folder",
        type=str,
        default="cngt_single_signs_512"
    )

    parser.add_argument(
        "--signbank_vocab_file",
        type=str,
        default="signbank_vocab.gzip"
    )

    parser.add_argument(
        "--framerate",
        type=int,
        default="25"
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default="16"
    )

    params, _ = parser.parse_known_args()
    main(params)
