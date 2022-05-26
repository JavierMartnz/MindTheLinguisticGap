import math
import torch
import torch.utils.data as data_utl
import numpy as np
import os
import cv2
from tqdm import tqdm

from src.utils.util import load_gzip, extract_zip


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(video_path, start_frame, window_size=64):
    frames = []
    # first, read the video frame by frame and store them in a list
    vcap = cv2.VideoCapture(video_path)
    while True:
        success, img = vcap.read()
        if not success:
            break
        w, h, c = img.shape
        # resize every frame to 256x256 and normalize them
        if w < 256 or h < 256:
            d = 256. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)

    # now make sure that the corresponding number of windows is filled
    last_frame = len(frames)
    if last_frame < start_frame + window_size:
        # iterate the number of missing frames to fill window
        for i in range(start_frame + window_size - last_frame):
            frames.append(frames[i])

    return np.asarray(frames[start_frame:start_frame + 64], dtype=np.float32)


# def load_flow_frames(root, vid, start, num):
#     frames = []
#     for i in range(start, start+num):
#         imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
#         imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
#
#     w,h = imgx.shape
#     if w < 224 or h < 224:
#         d = 224.-min(w,h)
#         sc = 1+d/min(w,h)
#         imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
#         imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
#
#     imgx = (imgx/255.)*2 - 1
#     imgy = (imgy/255.)*2 - 1
#     img = np.asarray([imgx, imgy]).transpose([1,2,0])
#     frames.append(img)
#     return np.asarray(frames, dtype=np.float32)

def make_dataset(cngt_zip, sb_zip, mode, class_encodings, split):
    assert split in {"train", "val", "test"}, "The splits can only have value 'train', 'val', and 'test'."

    num_classes = len(class_encodings)
    dataset = []

    # process zip files first
    if not os.path.isdir(cngt_zip[:-4]):
        cngt_extracted_root = extract_zip(cngt_zip)
    else:
        cngt_extracted_root = cngt_zip[:-4]
        print(f"{cngt_extracted_root} already exists, no need to extract")

    if not os.path.isdir(sb_zip[:-4]):
        sb_extracted_root = extract_zip(cngt_zip)
    else:
        sb_extracted_root = sb_zip[:-4]
        print(f"{sb_extracted_root} already exists, no need to extract")

    cngt_video_paths = [os.path.join(cngt_extracted_root, video) for video in os.listdir(cngt_extracted_root) if
                        video.endswith(".mpg")]
    sb_video_paths = [os.path.join(sb_extracted_root, video) for video in os.listdir(sb_extracted_root) if
                      video.endswith(".mp4")]

    # data splitting
    cngt_idx_train_val = int(len(cngt_video_paths) * (4 / 6))
    cngt_idx_val_test = int(len(cngt_video_paths) * (5 / 6))
    sb_idx_train_val = int(len(sb_video_paths) * (4 / 6))
    sb_idx_val_test = int(len(sb_video_paths) * (5 / 6))

    cngt_folds = {'train': cngt_video_paths[:cngt_idx_train_val],
                  'val': cngt_video_paths[cngt_idx_train_val:cngt_idx_val_test],
                  'test': cngt_video_paths[cngt_idx_val_test:]}

    sb_folds = {'train': sb_video_paths[:sb_idx_train_val],
                'val': sb_video_paths[sb_idx_train_val:sb_idx_val_test],
                'test': sb_video_paths[sb_idx_val_test:]}

    all_video_paths = cngt_folds[split]
    all_video_paths.extend(sb_folds[split])

    for video_path in tqdm(all_video_paths):
        metadata = load_gzip(video_path[:video_path.rfind(".m")] + ".gzip")
        num_frames = metadata.get("num_frames")
        if video_path.endswith(".mpg"):  # cngt video
            gloss_id = int(video_path.split("_")[-1][:-4])
        else:
            gloss_id = int(video_path.split("-")[-1][:-4])

        if mode == 'flow':
            num_frames = num_frames // 2

        label = np.zeros((num_classes, 64), np.float32)
        label_idx = class_encodings[gloss_id]
        for frame in range(64):
            label[label_idx, frame] = 1

        num_windows = math.ceil(num_frames / 64)

        for i in range(num_windows):
            dataset.append((video_path, label, num_frames, i * 64))

    return dataset


def get_class_encodings_from_zip(cngt_zip, sb_zip):
    # process zip files first
    if not os.path.isdir(cngt_zip[:-4]):
        cngt_extracted_root = extract_zip(cngt_zip)
    else:
        cngt_extracted_root = cngt_zip[:-4]
        print(f"{cngt_extracted_root} already exists, no need to extract")

    if not os.path.isdir(sb_zip[:-4]):
        sb_extracted_root = extract_zip(cngt_zip)
    else:
        sb_extracted_root = sb_zip[:-4]
        print(f"{sb_extracted_root} already exists, no need to extract")

    cngt_gloss_ids = [int(video.split("_")[-1][:-4]) for video in os.listdir(cngt_extracted_root) if video.endswith('.mpg')]
    sb_gloss_ids = [int(video.split("-")[-1][:-4]) for video in os.listdir(sb_extracted_root) if video.endswith('.mp4')]

    classes = list(set(cngt_gloss_ids).union(set(sb_gloss_ids)))

    class_to_idx = {}
    for i in range(len(classes)):
        class_to_idx[classes[i]] = i

    return class_to_idx


class I3Dataset(data_utl.Dataset):

    def __init__(self, cngt_zip, sb_zip, mode, split, window_size=64, transforms=None):
        self.mode = mode
        self.class_encodings = get_class_encodings_from_zip(cngt_zip, sb_zip)
        self.window_size = window_size
        self.transforms = transforms
        self.data = make_dataset(cngt_zip, sb_zip, mode, self.class_encodings, split)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        video_path, label, num_frames, start_frame = self.data[index]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(video_path, start_frame, self.window_size)
        # else:
        #     imgs = load_flow_frames(self.root, vid, start_frame)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)