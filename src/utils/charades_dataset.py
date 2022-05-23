import math

import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import random
import os
import os.path
import gzip
import cv2
import pickle


def load_gzip(filepath):
    with gzip.open(filepath, "rb") as f:
        loaded_object = pickle.load(f)
        f.close()
        return loaded_object


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


def load_rgb_frames(root, vid, start):
    frames = []
    vcap = cv2.VideoCapture(os.path.join(root, vid))
    while True:
        success, img = vcap.read()
        if not success:
            break
        w, h, c = img.shape
        if w < 256 or h < 256:
            d = 256. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)

    last_frame = len(frames)
    if last_frame < start + 64:
        for _ in range(start + 64 - last_frame):
            frames.append(frames[-1])

    frames = frames[start:start + 64]

    return np.asarray(frames, dtype=np.float32)

    # for i in range(start, start+num):
    #     img = cv2.imread(os.path.join(root, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    #     w, h, c = img.shape
    #     if w < 226 or h < 226:
    #         d = 226. - min(w, h)
    #         sc = 1 + d / min(w, h)
    #         img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
    #     img = (img/255.) * 2 - 1
    #     frames.append(img)

    # return np.asarray(frames, dtype=np.float32)


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

def make_dataset(root, mode, class_encodings, num_classes=157):
    dataset = []

    # CNGT videos
    videos = [video for video in os.listdir(root) if video.endswith(".mpg")]

    for vid in videos:
        metadata = load_gzip(os.path.join(root, vid[:-4] + ".gzip"))
        num_frames = metadata.get("num_frames")
        gloss_id = int(vid.split("_")[-1][:-4])

        if mode == 'flow':
            num_frames = num_frames // 2

        label = np.zeros((num_classes, 64), np.float32)
        label_idx = class_encodings[gloss_id]
        for frame in range(64):
            label[label_idx, frame] = 1

        num_windows = math.ceil(num_frames / 64)

        for i in range(num_windows):
            dataset.append((vid, label, num_frames, i * 64))

    return dataset

    # with open(split_file, 'r') as f:
    #     data = json.load(f)
    #
    # i = 0
    # for vid in data.keys():
    #     if data[vid]['subset'] != split:
    #         continue
    #
    #     if not os.path.exists(os.path.join(root, vid)):
    #         continue
    #
    #     num_frames = len(os.listdir(os.path.join(root, vid)))
    #     if mode == 'flow':
    #         num_frames = num_frames//2
    #
    #     if num_frames < 66:
    #         continue
    #
    #     label = np.zeros((num_classes,num_frames), np.float32)
    #
    #     fps = num_frames/data[vid]['duration']
    #     for ann in data[vid]['actions']:
    #         for fr in range(0,num_frames,1):
    #             if ann[1] < fr/fps < ann[2]:
    #                 label[ann[0], fr] = 1 # binary classification
    #     dataset.append((vid, label, data[vid]['duration'], num_frames))
    #     i += 1
    #
    # return dataset


class Charades(data_utl.Dataset):

    def __init__(self, root, mode, class_encodings, transforms=None):
        self.data = make_dataset(root, mode, class_encodings, num_classes=len(class_encodings))
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.class_encodings = class_encodings

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, num_frames, start_frame = self.data[index]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_frame)
        # else:
        #     imgs = load_flow_frames(self.root, vid, start_frame)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
