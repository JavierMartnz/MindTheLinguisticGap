import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from zipfile import ZipFile
from torchvision.io import read_video
from tqdm import tqdm

from src.utils.transforms import im_color_jitter, color_normalize
from src.utils.my_collate import my_collate
from src.utils.util import load_gzip

def count_occurrences(my_list):
    # Creating an empty dictionary
    count = {}
    for i in my_list:
        count[i] = count.get(i, 0) + 1
    return count

def count_video_frames(video_path):
    n_frames = 0
    vcap = cv2.VideoCapture(video_path)
    while True:
        status, frame = vcap.read()
        if not status:
            break
        n_frames += 1
    return n_frames


def get_class_encodings(cngt_clips_path, signbank_path):
    cngt_clips_path = cngt_clips_path[:cngt_clips_path.rfind(".zip")]
    signbank_path = signbank_path[:signbank_path.rfind(".zip")]

    # cngt_gloss_ids = [int(file.split("_")[-1][:-4]) for file in os.listdir(cngt_clips_path) if file.endswith('.mpg')]
    cngt_gloss_ids = {}
    signbank_gloss_ids = [int(file.split("-")[-1][:-4]) for file in os.listdir(signbank_path) if file.endswith('.mp4')]

    classes = list(set(cngt_gloss_ids).union(set(signbank_gloss_ids)))

    class_to_idx = {}
    for i in range(len(classes)):
        class_to_idx[classes[i]] = i

    return class_to_idx


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


class I3DDataset(Dataset):
    def __init__(self, window_size, transforms=None):
        self.samples = []
        self.window_size = window_size
        self.class_encoding = {}
        self.transforms = transforms

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        """
        sample = self.samples[index]
        start_frame = int(sample.get("start_frame"))
        last_window_frame = start_frame + self.window_size

        vcap = cv2.VideoCapture(sample.get("video_path"))
        video = []
        num_frames = 0
        success, bgr = vcap.read()
        while success:
            rgb = bgr[:, :, [2, 1, 0]]
            video.append(rgb)  # [T, H, W, C]
            num_frames += 1
            success, bgr = vcap.read()

        x = np.empty([self.window_size, 256, 256, 3], dtype=np.float32)

        for i in range(self.window_size):
            if start_frame + i < num_frames:
                x[i, :, :, :] = video[start_frame + i]
            else:  # this means we need to fill the window
                x[i, :, :, :] = x[i - 1, :, :, :]

        # x = (x / 255).astype(np.float32)  # normalize pixels to value in [0,1]

        num_classes = len(self.class_encoding)
        label = np.zeros(num_classes, dtype=np.float32)
        label[self.class_encoding[int(sample.get("gloss_id"))]] = 1

        if self.transforms:
            x = self.transforms(x)

        x = np.transpose(x, (3, 0, 1, 2))  # [C, T, H, W] for i3d

        return torch.from_numpy(x), torch.from_numpy(label)


def load_data(data_cfg: dict, set_names: list, transforms: list) -> list:
    allowed_sets = {'train', 'val', 'test'}

    cgnt_path = data_cfg.get('cngt_clips_path')
    signbank_path = data_cfg.get('signbank_path')
    window_size = data_cfg.get("window_size")

    assert len(transforms) == len(set_names), "You must give a list of transforms with the same length as the folds" \
                                              "required. If you want no transforms, pass [None, ...] as argument"

    datasets = {}
    for i, set_name in enumerate(set_names):
        assert set_name in allowed_sets, f"The set names must be in {allowed_sets}"
        datasets[set_name] = I3DDataset(window_size, transforms[i])

    if not os.path.isdir(cgnt_path[:-4]):
        extracted_videos_root = extract_zip(cgnt_path)
    else:
        extracted_videos_root = cgnt_path[:-4]
        print(f"{extracted_videos_root} already exists, no need to extract")

    cngt_videos = [file for file in os.listdir(extracted_videos_root) if file.endswith('.mpg')]
    cngt_metadata = [file for file in os.listdir(extracted_videos_root) if file.endswith('.gzip')]

    assert len(cngt_videos) == len(cngt_metadata), "CNGT videos and metadata are unmatched. Please check again that" \
                                                   "every video has an associated .gzip file"

    print(f"Loading videos from CNGT clips")
    if cngt_videos:
        video_paths = [os.path.join(extracted_videos_root, video) for video in cngt_videos]
        gloss_ids = [int(video.split("_")[-1][:-4]) for video in cngt_videos]  # save the id of the gloss

        # # This piece of code gets only the clips for a gloss that appears at least 3 times in the CNGT
        # glosses = [video.split("_")[-2] for video in cngt_videos]
        # gloss_occ = count_occurrences(glosses)
        # video_flags = [0] * len(glosses)
        # # flag where the video needs to be used, based on the number of occurences of a gloss
        # for i, gloss in enumerate(glosses):
        #     if gloss_occ[gloss] >= 3:
        #         video_flags[i] = 1
        #
        # video_paths = [video_paths[i] for i in range(len(video_paths)) if video_flags[i] == 1]

        split_idx_train_val = int(len(video_paths) * (4 / 6))
        split_idx_val_test = int(len(video_paths) * (5 / 6))

        fold_idxs = range(len(video_paths))
        folds = {'train': fold_idxs[:split_idx_train_val],
                 'val': fold_idxs[split_idx_train_val:split_idx_val_test],
                 'test': fold_idxs[split_idx_val_test:]}

        # make sure we only iterate over the wanted idxs
        wanted_idxs = []
        for set_name in set_names:
            wanted_idxs.extend(folds[set_name])

        for i in tqdm(range(len(wanted_idxs))):

            metadata = load_gzip(os.path.join(extracted_videos_root, cngt_metadata[i]))
            n_frames = metadata.get("num_frames")

            # this will work even for videos that are smaller than the window size
            for set_name in set_names:
                if i in folds[set_name]:
                    for start_frame in metadata.get("start_frames"):
                        datasets[set_name].samples.append({"video_path": video_paths[i], "gloss_id": gloss_ids[i],
                                                           "start_frame": start_frame, "num_frames": n_frames})
                    break

    if not os.path.isdir(signbank_path[:-4]):
        extracted_signbank_root = extract_zip(signbank_path)
    else:
        extracted_signbank_root = signbank_path[:-4]
        print(f"{extracted_signbank_root} already exists, no need to extract")

    signbank_videos = [file for file in os.listdir(extracted_signbank_root) if file.endswith('.mp4')]
    signbank_metadata = [file for file in os.listdir(extracted_signbank_root) if file.endswith('.gzip')]

    assert len(signbank_videos) == len(signbank_metadata), "Signbank videos and metadata are unmatched. Please check" \
                                                           "again that every video has an associated .gzip file"

    print(f"Loading videos from Signbank")
    if signbank_videos:
        video_paths = [os.path.join(extracted_signbank_root, video) for video in signbank_videos]
        gloss_ids = [int(video.split("-")[-1][:-4]) for video in signbank_videos]

        split_idx_train_val = int(len(video_paths) * (4 / 6))
        split_idx_val_test = int(len(video_paths) * (5 / 6))

        fold_idxs = range(len(video_paths))
        folds = {'train': fold_idxs[:split_idx_train_val],
                 'val': fold_idxs[split_idx_train_val:split_idx_val_test],
                 'test': fold_idxs[split_idx_val_test:]}

        # make sure we only iterate over the wanted idxs
        wanted_idxs = []
        for set_name in set_names:
            wanted_idxs.extend(folds[set_name])

        for i in tqdm(range(len(wanted_idxs))):
            metadata = load_gzip(os.path.join(extracted_signbank_root, signbank_metadata[i]))
            n_frames = metadata.get("num_frames")

            # this will work even for videos that are smaller than the window size
            for set_name in set_names:
                if i in folds[set_name]:
                    for start_frame in metadata.get("start_frames"):
                        datasets[set_name].samples.append({"video_path": video_paths[i], "gloss_id": gloss_ids[i],
                                                           "start_frame": start_frame, "num_frames": n_frames})
                    break

    for k in datasets.keys():
        datasets[k].class_encoding = get_class_encodings(cgnt_path, signbank_path)

    return list(datasets.values())
