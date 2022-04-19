import os
import argparse
from multiprocessing import Pool
from zipfile import ZipFile
from pathlib import Path
from tqdm import tqdm
import math

from src.utils.util import save_gzip, count_video_frames

def resize_video(video_path, output_root, window_size):
    filename = os.path.basename(video_path)
    output_filename = os.path.join(output_root, filename)
    cmd = f'ffmpeg -hide_banner -loglevel error -i {video_path} -y -vf "scale=256:256" -r 25 {output_filename}'
    os.system(cmd)

    # save the metadata for data loading
    num_frames = count_video_frames(output_filename)
    metadata = {"num_frames": num_frames, "start_frames": []}
    num_clips = math.ceil(num_frames / window_size)
    for j in range(num_clips):
        metadata["start_frames"].append(j * window_size)

    save_gzip(metadata, output_filename[:-3] + 'gzip')

def main(params):
    dataset_root = params.dataset_root
    output_root = params.output_root
    window_size = params.window_size

    os.makedirs(output_root, exist_ok=True)

    pool = Pool()

    print(f"Resizing SignBank videos from \n{dataset_root}\nto\n{output_root}")

    for subdir, _, files in os.walk(dataset_root):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(subdir, file)
                pool.apply_async(resize_video, args=(video_path, output_root, int(window_size)))

    pool.close()
    pool.join()

    print("Zipping SignBank resized videos")

    zip_basedir = Path(output_root).parent
    zip_filename = os.path.basename(output_root) + '.zip'
    all_filenames = os.listdir(output_root)

    with ZipFile(os.path.join(zip_basedir, zip_filename), 'w') as zipfile:
        for filename in tqdm(all_filenames):
            zipfile.write(os.path.join(output_root, filename), filename)

    if os.path.isfile(os.path.join(output_root, zip_filename)):
        # maybe remove in a future
        print("Zipfile saved succesfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="D:/Thesis/datasets/NGT_Signbank"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="D:/Thesis/datasets/NGT_Signbank_resized"
    )

    parser.add_argument(
        "--window_size",
        type=str,
        default="64"
    )

    params, _ = parser.parse_known_args()
    main(params)
