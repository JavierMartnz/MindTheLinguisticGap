import os
from tqdm import tqdm
import argparse
import pympi
from multiprocessing import Pool
from pathlib import Path
from zipfile import ZipFile
import math
import shutil
import sys

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

from src.utils.util import save_gzip, count_video_frames, extract_zip


def resize_video(video_path, output_root, video_size, framerate, window_size=None, is_sb=False):
    # .mpeg doesn't support a framerate lower than 24, so change to .mov if that happens
    if framerate < 24:
        filename = os.path.basename(video_path).replace("mpg", "mov")
    else:
        filename = os.path.basename(video_path)
    output_filename = os.path.join(output_root, filename)

    # if the file doesn't exist already
    if not os.path.exists(output_filename):
        if is_sb:
            assert type(window_size) is int, "Please enter a valid window size"

            cmd = f'ffmpeg -hide_banner -loglevel error -i {video_path} -y -vf "scale={video_size}:{video_size}" -r {framerate} -preset ultrafast {output_filename}'
            os.system(cmd)

            # save the metadata for data loading
            num_frames = count_video_frames(output_filename)
            metadata = {"num_frames": num_frames, "start_frames": []}
            num_clips = math.ceil(num_frames / window_size)
            for j in range(num_clips):
                metadata["start_frames"].append(j * window_size)

            save_gzip(metadata, output_filename[:-3] + 'gzip')
        else:
            cmd = f'ffmpeg -hwaccel cuda -hide_banner -loglevel error -i {video_path} -y -vf "scale={video_size}:{video_size}" -r {framerate} -preset ultrafast {output_filename}'
            os.system(cmd)

            # also copy the annotation file to the output folder
            ann_path = video_path[:video_path.rfind(".mpg")] + ".eaf"
            ann_dest_path = os.path.join(output_root, os.path.basename(ann_path))
            shutil.copy(ann_path, ann_dest_path)


def main(params):
    # read user arguments
    root = params.root
    cngt_folder = params.cngt_folder
    cngt_output_folder = params.cngt_output_folder
    sb_folder = params.sb_folder
    sb_output_folder = params.sb_output_folder
    video_size = params.video_size
    framerate = params.framerate
    window_size = params.window_size

    # build the entire paths for the datasets
    cngt_root = os.path.join(root, cngt_folder)
    cngt_output_root = os.path.join(root, cngt_output_folder)
    sb_root = os.path.join(root, sb_folder)
    sb_output_root = os.path.join(root, sb_output_folder)

    if not os.path.exists(cngt_root):
        # if the specified folder doesn't exist, check for zip equivalent and unzip
        cngt_zip = cngt_root + '.zip'
        if os.path.exists(cngt_zip):
            extract_zip(cngt_zip)

    if not os.path.exists(sb_root):
        sb_zip = sb_root + '.zip'
        if os.path.exists(sb_zip):
            extract_zip(sb_zip)

    cngt_videos = [file for file in os.listdir(cngt_root) if file.endswith('.mpg')]
    os.makedirs(cngt_output_root, exist_ok=True)

    print(f"The specified video resolution is {video_size}x{video_size} px at {framerate} fps.")

    print(f"Resizing CNGT videos from \n{cngt_root}\nto\n{cngt_output_root}")

    # multiprocessing bit based on https://github.com/tqdm/tqdm/issues/484
    pool = Pool()
    pbar = tqdm(total=len(cngt_videos))

    def update(*a):
        pbar.update()

    for i in range(pbar.total):
        pool.apply_async(resize_video,
                         args=(os.path.join(cngt_root, cngt_videos[i]),
                               cngt_output_root,
                               video_size,
                               framerate),
                         callback=update)

    pool.close()
    pool.join()

    print(f"Resizing SignBank videos from \n{sb_root}\nto\n{sb_output_root}")

    os.makedirs(sb_output_root, exist_ok=True)

    pool = Pool()

    for subdir, _, files in os.walk(sb_root):
        for file in files:
            if file.endswith(".mp4"):
                pool.apply_async(resize_video,
                                 args=(os.path.join(subdir, file),
                                       sb_output_root,
                                       video_size,
                                       framerate,
                                       window_size,
                                       True))

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
        default="CNGT_isolated_signers"
    )

    parser.add_argument(
        "--cngt_output_folder",
        type=str,
        default="CNGT_12fps"
    )

    parser.add_argument(
        "--sb_folder",
        type=str,
        default="NGT_Signbank"
    )

    parser.add_argument(
        "--sb_output_folder",
        type=str,
        default="NGT_Signbank_12fps"
    )

    parser.add_argument(
        "--video_size",
        type=int,
        default="256"
    )

    parser.add_argument(
        "--framerate",
        type=int,
        default="12"
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default="16"
    )

    params, _ = parser.parse_known_args()
    main(params)
