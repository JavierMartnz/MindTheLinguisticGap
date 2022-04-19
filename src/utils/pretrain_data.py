import math
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
from zipfile import ZipFile
from torchvision.io import read_video
from tqdm import tqdm

from src.utils.transforms import im_color_jitter, color_normalize
from src.utils.my_collate import my_collate
from src.utils.util import load_gzip


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

    cngt_gloss_ids = [int(file.split("_")[-1][:-4]) for file in os.listdir(cngt_clips_path) if file.endswith('.mpg')]
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


# def make_data_loader(dataset: Dataset, batch_size: int,  shuffle: bool = False) -> DataLoader:
#     params = {
#         'batch_size': batch_size,
#         'shuffle': shuffle,
#         'num_workers': 4,
#         'collate_fn': my_collate}
#     return DataLoader(dataset, **params)

class I3DDataset(Dataset):
    def __init__(self, window_size):
        self.samples = []
        self.window_size = window_size
        self.class_encoding = {}

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
        num_frames = int(sample.get("num_frames"))
        start_frame = int(sample.get("start_frame"))
        last_window_frame = start_frame + self.window_size

        if last_window_frame > num_frames:
            # we pad with copies of last frame
            X_raw, _, _ = read_video(filename=sample["video_path"])  # returns Tensor[T, H, W, C]
            X = torch.empty([self.window_size, X_raw.size(1), X_raw.size(2), X_raw.size(3)])
            X[:(num_frames - start_frame), :, :, :] = X_raw[start_frame:, :, :, :]
            for i in range((num_frames - start_frame), self.window_size):
                X[i, :, :, :] = X[i - 1, :, :, :]

        else:
            X_raw, _, _ = read_video(filename=sample["video_path"])  # returns Tensor[T, H, W, C]
            X = X_raw[start_frame:last_window_frame, :, :, :]

        X = X.permute(3, 0, 1, 2)  # Tensor[C, T, H, W] as needed by i3d
        Y = torch.zeros(len(self.class_encoding))
        Y[self.class_encoding[int(sample.get("gloss_id"))]] = int(sample.get("gloss_id"))
        # Y = torch.zeros(len(self.class_encoding), self.window_size)
        # Y[self.class_encoding[int(sample.get("gloss_id"))], :] = int(sample.get("gloss_id"))

        return X, Y


def load_data(data_cfg: dict, set_names: list) -> list[Dataset]:
    '''

    Args:
        data_cfg:
        set_names:

    Returns:

    '''
    allowed_sets = {'train', 'val', 'test'}

    cgnt_path = data_cfg.get('cngt_clips_path')
    signbank_path = data_cfg.get('signbank_path')
    window_size = data_cfg.get("window_size")

    datasets = {}
    for set_name in set_names:
        assert set_name in allowed_sets, f"The set names must be in {allowed_sets}"
        datasets[set_name] = I3DDataset(window_size)

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
