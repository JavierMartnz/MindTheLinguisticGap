import math
import random

import torch
import torch.utils.data as data_utl
import numpy as np
import os
import cv2
from tqdm import tqdm

from src.utils.util import load_gzip, extract_zip


def count_occurrences(my_list) -> dict:
    """
    Given a list, return a dictionary with unique values of the list as keys and their number of occurences in the list as values
    """
    # Creating an empty dictionary
    count = {}
    for i in my_list:
        count[i] = count.get(i, 0) + 1
    return count


def filter_top_glosses(gloss_ids: list, k: int, top_gloss_ids=None) -> list:
    """
    Given a list of gloss IDs, return the top k most frequent IDs. If 'top_gloss_ids' is given, filter the input 'gloss_ids' based on its
    content rather than counting the most frequent.

    Args:
        gloss_ids: list of gloss ids to be filtered
        k: the number of most frequent ids to be returned
        top_gloss_ids: list containing ids to filter with
    Returns:
    """
    if top_gloss_ids:
        assert len(top_gloss_ids) == k, "The shape of top_gloss_ids does not match k"
    # if the top gloss ids are not given, then calculate them based on the content of the input list
    else:
        id_occ = count_occurrences(gloss_ids)
        sorted_id_occ = dict(sorted(id_occ.items(), key=lambda item: item[1], reverse=True))
        top_gloss_ids = list(sorted_id_occ.keys())[:k]

    # binary array containing the positions of glosses that made the cut
    video_flags = [0] * len(gloss_ids)
    for i, gloss in enumerate(gloss_ids):
        if gloss in top_gloss_ids:
            video_flags[i] = 1

    return video_flags, top_gloss_ids


def video_to_tensor(pic):
    """
    Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(video_path: str, start_frame: int, window_size=64):
    """
    Given a path to a video file, load T rgb frames given by 'window_size', starting from frame 'start_frame'

    Args:
        video_path: full path to video file
        start_frame: number of starting frame
        window_size: size of the wanted rgb frame tensor
    Returns: float32 Tensor of shape [T, W, H, C]
    """
    frames = []
    vcap = cv2.VideoCapture(video_path)
    # to save some reading time, set initial frame to the given start frame
    vcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while len(frames) < window_size:
        success, img = vcap.read()
        if not success:
            break
        w, h, c = img.shape
        # resize every frame to 256x256 and normalize them
        if w < 256 or h < 256:
            d = 256. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = img / 255.  # normalize values to range [0, 1]
        img = img[:, :, [2, 1, 0]]  # opencv uses bgr so switch to rgb
        frames.append(img)

    assert len(frames) <= window_size, "ERROR: more frames than window size"

    # if the number of frames didn't fill the window, we loop the video from the start until the window is filled
    if len(frames) < window_size:
        for i in range(window_size - len(frames)):
            frames.append(frames[i])

    return torch.Tensor(np.asarray(frames, dtype=np.float32))


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

def build_dataset(cngt_zip: str, sb_zip: str, cngt_vocab_path: str, sb_vocab_path: str, mode: str, class_encodings: dict, window_size: int, split: str) -> list:
    assert split in {"train", "val", "test"}, "The variable 'split' can only have value  'train', 'val', and 'test'."

    num_classes = len(class_encodings)
    classes = list(class_encodings.keys())
    dataset = []

    # extract zip files (if not extracted already) and get the directory root
    if not os.path.isdir(cngt_zip[:-4]):
        cngt_extracted_root = extract_zip(cngt_zip)
    else:
        cngt_extracted_root = cngt_zip[:-4]

    if not os.path.isdir(sb_zip[:-4]):
        sb_extracted_root = extract_zip(cngt_zip)
    else:
        sb_extracted_root = sb_zip[:-4]

    cngt_videos = [file for file in os.listdir(cngt_extracted_root) if file.endswith('.mpg')]
    sb_videos = [file for file in os.listdir(sb_extracted_root) if file.endswith('.mp4')]

    # only load the paths that correspond to filtered glosses and attribute it to a given signer
    # cngt_video_paths = [os.path.join(cngt_extracted_root, video) for i, video in enumerate(cngt_videos) if
    #                     int(video.split("_")[-1][:-4]) in classes]
    sb_video_paths = [os.path.join(sb_extracted_root, video) for i, video in enumerate(sb_videos) if
                      int(video.split("-")[-1][:-4]) in classes]

    # create dictionary of each signer with its video paths
    n_paths = 0
    cngt_signer_paths = {}
    for file in os.listdir(cngt_extracted_root):
        # filter videos based on the selected glosses
        if file.endswith('.mpg') and int(file.split("_")[-1][:-4]) in classes:
            n_paths += 1
            signer = os.path.basename(file).split('_')[0]
            signer_paths = cngt_signer_paths.get(signer, -1)
            if signer_paths == -1:  # if the signer wasn't added yet
                signer_paths = [os.path.join(cngt_extracted_root, file)]
            else:
                signer_paths.append(os.path.join(cngt_extracted_root, file))

            cngt_signer_paths[signer] = signer_paths

    signers = list(cngt_signer_paths.keys())

    print(f"Theres a total of {len(signers)} signers and {n_paths} clips")

    # randomize order of the signers
    random.seed(42)
    random.shuffle(signers)

    train_size = int(n_paths * (4/6))
    val_test_size = int(n_paths * (1/6))

    cngt_folds = {'train': [], 'val': [], 'test': []}

    # loop for the stratification of the data splits
    for signer in signers:
        # if the training split isn't full
        if len(cngt_folds['train']) < train_size:
            # if it can still fit the next whole signer
            if len(cngt_folds['train']) + len(cngt_signer_paths[signer]) <= train_size:
                cngt_folds['train'].extend(cngt_signer_paths[signer])
            # if it can't, then fill with paths from the next signer
            else:
                paths_to_fill = train_size - len(cngt_folds['train'])
                cngt_folds['train'].extend(cngt_signer_paths[signer][:paths_to_fill])
                # fill beginning of val split
                cngt_folds['val'].extend(cngt_signer_paths[signer][paths_to_fill:])

        # if the val split isn't full
        elif len(cngt_folds['val']) < val_test_size:
            if len(cngt_folds['val']) + len(cngt_signer_paths[signer]) <= val_test_size:
                cngt_folds['val'].extend(cngt_signer_paths[signer])
            # if it can't, then fill with paths from the next signer
            else:
                paths_to_fill = val_test_size - len(cngt_folds['val'])
                cngt_folds['val'].extend(cngt_signer_paths[signer][:paths_to_fill])
                # fill beginning of val split
                cngt_folds['test'].extend(cngt_signer_paths[signer][paths_to_fill:])

        # if code gets here, fill test split with the remaining paths
        elif len(cngt_folds['test']) < val_test_size:
            cngt_folds['test'].extend(cngt_signer_paths[signer])

    cnt_dict = {}
    # print the videos per signer count for every split
    for split in ['train', 'val', 'test']:
        cnt_dict[split] = count_occurrences([os.path.basename(path).split('_')[0] for path in cngt_folds[split]])
        print(f"The {split} split has {len(cnt_dict[split].keys())} different signers and a total of {sum(cnt_dict[split].values())} clips")

    print(f"train-val overlap: {set(cnt_dict['train'].keys()).intersection(set(cnt_dict['val'].keys()))}")
    print(f"val-test overlap: {set(cnt_dict['val'].keys()).intersection(set(cnt_dict['test'].keys()))}")

    # signers are not indicated in signbank, therefore we just do a normal random split
    random.shuffle(sb_video_paths)

    # data splitting train:val:test with ratio 4:1:1
    sb_idx_train_val = int(len(sb_video_paths) * (4 / 6))
    sb_idx_val_test = int(len(sb_video_paths) * (5 / 6))

    sb_folds = {'train': sb_video_paths[:sb_idx_train_val],
                'val': sb_video_paths[sb_idx_train_val:sb_idx_val_test],
                'test': sb_video_paths[sb_idx_val_test:]}

    all_video_paths = cngt_folds[split]
    # THIS NEXT LINE IS ONLY FOR TESTING, SHOULD BE REMOVED
    # all_video_paths = []
    all_video_paths.extend(sb_folds[split])

    label_dict = {}

    for video_path in tqdm(all_video_paths):
        metadata = load_gzip(video_path[:video_path.rfind(".m")] + ".gzip")
        num_frames = metadata.get("num_frames")
        if video_path.endswith(".mpg"):  # cngt video
            gloss_id = int(video_path.split("_")[-1][:-4])
        else:
            gloss_id = int(video_path.split("-")[-1][:-4])

        if mode == 'flow':
            num_frames = num_frames // 2

        label = np.zeros((num_classes, window_size), np.float32)
        label_idx = class_encodings[gloss_id]

        # TEST
        # get glosses from the class encodings
        cngt_vocab = load_gzip(cngt_vocab_path)
        sb_vocab = load_gzip(sb_vocab_path)
        # join cngt and sb vocabularies (gloss to id dictionary)
        sb_vocab.update(cngt_vocab)
        gloss_to_id = sb_vocab['gloss_to_id']

        gloss = list(gloss_to_id.keys())[list(gloss_to_id.values()).index(gloss_id)]

        if gloss not in label_dict.keys():
            label_dict[gloss] = 0

        for frame in range(window_size):
            label[label_idx, frame] = 1

        num_windows = math.ceil(num_frames / window_size)

        for i in range(num_windows):
            # TEST
            label_dict[gloss] += 1
            dataset.append((video_path, label, num_frames, i * window_size))

    print(f"The labels and label count is {label_dict}")

    return dataset


def get_class_encodings_from_zip(cngt_zip, sb_zip, filter_num=None):
    # process zip files first
    if not os.path.isdir(cngt_zip[:-4]):
        cngt_extracted_root = extract_zip(cngt_zip)
    else:
        cngt_extracted_root = cngt_zip[:-4]
        # print(f"{cngt_extracted_root} already exists, no need to extract")

    if not os.path.isdir(sb_zip[:-4]):
        sb_extracted_root = extract_zip(cngt_zip)
    else:
        sb_extracted_root = sb_zip[:-4]
        # print(f"{sb_extracted_root} already exists, no need to extract")

    cngt_gloss_ids = [int(video.split("_")[-1][:-4]) for video in os.listdir(cngt_extracted_root) if video.endswith('.mpg')]
    sb_gloss_ids = {}

    if filter_num is None:  # default case
        sb_gloss_ids = [int(video.split("-")[-1][:-4]) for video in os.listdir(sb_extracted_root) if video.endswith('.mp4')]
    else:
        assert type(filter_num) == int, "The variable 'filter_num' must be an integer"
        _, top_cngt_ids = filter_top_glosses(cngt_gloss_ids, filter_num)
        cngt_gloss_ids = top_cngt_ids

    # THE NEXT 3 LINES ARE ONLY FOR TESTING, SHOULD BE REMOVED
    # cngt_gloss_ids = {}
    # sb_gloss_ids = [int(video.split("-")[-1][:-4]) for video in os.listdir(sb_extracted_root) if video.endswith('.mp4')]
    # _, sb_gloss_ids = filter_top_glosses(sb_gloss_ids, filter_num)

    classes = list(set(cngt_gloss_ids).union(set(sb_gloss_ids)))

    class_to_idx = {}
    for i in range(len(classes)):
        class_to_idx[classes[i]] = i

    return class_to_idx


class I3Dataset(data_utl.Dataset):

    def __init__(self, cngt_zip, sb_zip, cngt_vocab_path, sb_vocab_path, mode, split, window_size=64, transforms=None, filter_num=None):
        self.mode = mode
        self.class_encodings = get_class_encodings_from_zip(cngt_zip, sb_zip, filter_num)
        self.window_size = window_size
        self.transforms = transforms
        self.data = build_dataset(cngt_zip, sb_zip, cngt_vocab_path, sb_vocab_path, mode, self.class_encodings, window_size, split)

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

        # pytorch transforms take input tensor of shape [C, T, H, W]
        imgs = imgs.permute([3, 0, 1, 2])

        if self.transforms:
            imgs = self.transforms(imgs)

        return imgs, torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
