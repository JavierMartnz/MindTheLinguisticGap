import pickle
import gzip
import os
import cv2
from zipfile import ZipFile

def save_gzip(obj, filepath):
    with gzip.open(filepath, "wb") as f:
        pickle.dump(obj, f, -1)
        f.close()


def load_gzip(filepath):
    with gzip.open(filepath, "rb") as f:
        loaded_object = pickle.load(f)
        f.close()
        return loaded_object

def count_video_frames(video_path):
    """
    Counts the number of frames in a video by iterating through the video itself. Arcaic but reliable.
    """
    n_frames = 0
    vcap = cv2.VideoCapture(video_path)
    while True:
        status, frame = vcap.read()
        if not status:
            break
        n_frames += 1
    return n_frames

def extract_zip(zip_path):
    if not os.path.isfile(zip_path):
        print(f"{zip_path} does not exist")
        return
    if not zip_path.endswith("zip"):
        print(f"{zip_path} is not a zip file")
        return

    data_root = os.path.dirname(zip_path)
    extracted_dir = os.path.basename(zip_path)[:-4]
    extracted_root = os.path.join(data_root, extracted_dir)

    print(f"Extracting zipfile from {zip_path} to {extracted_root}")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_root)
    print("Extraction successful!")

    return extracted_root