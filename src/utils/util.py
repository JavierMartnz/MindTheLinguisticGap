import pickle
import gzip
import os
import cv2

def save_gzip(obj, filepath):
    with gzip.open(filepath, "wb") as f:
        pickle.dump(obj, f, -1)
        f.close()


def load_gzip(filepath):
    with gzip.open(filepath, "rb") as f:
        loaded_object = pickle.load(f)
        f.close()
        return loaded_object


def make_dir(dir: str) -> str:
    if not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def save_vocab(gloss_to_id_dict, vocab_path):
    save_data = {'glosses': list(gloss_to_id_dict.keys()), 'gloss_to_id': gloss_to_id_dict}
    vocab_dir = os.path.dirname(vocab_path)
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    save_gzip(save_data, vocab_path)
    print(f'Vocab saved at {vocab_path}')

def count_video_frames(video_path):
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