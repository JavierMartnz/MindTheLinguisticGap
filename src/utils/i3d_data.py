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


def filter_top_glosses(gloss_ids: list, k: int, top_gloss_ids=None) -> tuple:
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
    if start_frame != 0:
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


def build_stratified_dataset(cngt_video_paths: list, sb_video_paths: list, sb_vocab: dict, mode: str,
                             class_encodings: dict, window_size: int, split: str, random_seed: int) -> list:
    classes = list(class_encodings.keys())
    gloss_to_id = sb_vocab["gloss_to_id"]

    # CREATE DICTIONARY OF VIDEO PATHS FOR EACH SIGNER
    n_paths = 0
    cngt_signer_dicts = {}
    class_cnt_dict = {}

    for cngt_video_path in cngt_video_paths:
        if int(cngt_video_path.split("_")[-1][:-4]) in classes:
            n_paths += 1
            metadata = load_gzip(cngt_video_path[:cngt_video_path.rfind(".m")] + ".gzip")
            num_frames = metadata.get("num_frames")

            gloss_id = int(cngt_video_path.split("_")[-1][:-4])
            gloss = list(gloss_to_id.keys())[list(gloss_to_id.values()).index(gloss_id)]

            # count number of frames taking into account window expansion
            if gloss not in class_cnt_dict.keys():
                class_cnt_dict[gloss] = 0

            num_windows = math.ceil(num_frames / window_size)
            class_cnt_dict[gloss] += window_size * num_windows

            signer = os.path.basename(cngt_video_path).split('_')[0]
            signer_dict = cngt_signer_dicts.get(signer, -1)

            # if the signer wasn't added yet, initialize dictionary
            if signer_dict == -1:
                cngt_signer_dicts[signer] = {}

            if gloss not in cngt_signer_dicts[signer].keys():  # if the signer doesn't have the gloss added yet
                cngt_signer_dicts[signer][gloss] = [(cngt_video_path, num_frames)]
            else:
                cngt_signer_dicts[signer][gloss].append((cngt_video_path, num_frames))

    signers = list(cngt_signer_dicts.keys())

    print(f"Theres a total of {len(signers)} signers and {n_paths} videos")

    total_num_frames = sum(class_cnt_dict.values())
    train_size = int(total_num_frames * (4 / 6))
    val_test_size = int(total_num_frames * (1 / 6))

    glosses = class_cnt_dict.keys()

    # get the frames' ratio for each class to respect it when creating the splits
    ratio_dict = {key: class_cnt_dict[key] / total_num_frames for key in glosses}

    # calculate the actual size of the splits for each class wrt the window size
    train_size_dict = {key: int(train_size * ratio_dict[key] // window_size) for key in glosses}
    val_test_size_dict = {key: int(val_test_size * ratio_dict[key] // window_size) for key in glosses}

    # shuffle the signers for random assignment
    random.seed(random_seed)
    random.shuffle(signers)

    cngt_split_dict = {'train': [], 'val': [], 'test': []}
    windows_filled_dict = {split: {gloss: 0 for gloss in glosses} for split in {'train', 'val', 'test'}}

    for signer in signers:
        for gloss in cngt_signer_dicts[signer].keys():

            # iterate over all paths
            while len(cngt_signer_dicts[signer][gloss]) > 0:
                path, num_frames = cngt_signer_dicts[signer][gloss].pop()
                num_windows = math.ceil(num_frames / window_size)

                # if the split is not filled yet, append the video path
                if windows_filled_dict['train'][gloss] < train_size_dict[gloss]:
                    cngt_split_dict['train'].append(path)
                    windows_filled_dict['train'][gloss] += num_windows
                elif windows_filled_dict['val'][gloss] < val_test_size_dict[gloss]:
                    cngt_split_dict['val'].append(path)
                    windows_filled_dict['val'][gloss] += num_windows
                else:
                    cngt_split_dict['test'].append(path)
                    windows_filled_dict['test'][gloss] += num_windows

    cnt_dict = {}
    # print the videos per signer count for every split
    for t_split in ['train', 'val', 'test']:
        cnt_dict[t_split] = count_occurrences([os.path.basename(path).split('_')[0] for path in cngt_split_dict[t_split]])
        print(f"The {t_split} split has {len(cnt_dict[t_split].keys())} different signers and a total of {sum(cnt_dict[t_split].values())} clips")

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

    all_video_paths = cngt_split_dict[split]
    all_video_paths.extend(sb_folds[split])

    num_classes = len(class_encodings)
    dataset = []
    label_dict = {}

    for video_path in tqdm(all_video_paths):
        metadata = load_gzip(video_path[:video_path.rfind(".m")] + ".gzip")
        num_frames = metadata.get("num_frames")
        if video_path.endswith(".mpg") or video_path.endswith(".mov"):  # cngt video
            gloss_id = int(video_path.split("_")[-1][:-4])
        else:
            gloss_id = int(video_path.split("-")[-1][:-4])

        # if mode == 'flow':
        #     num_frames = num_frames // 2

        label = np.zeros(num_classes, np.float32)
        label_idx = class_encodings[gloss_id]
        gloss = list(gloss_to_id.keys())[list(gloss_to_id.values()).index(gloss_id)]

        if gloss not in label_dict.keys():
            label_dict[gloss] = 0

        label[label_idx] = 1

        num_windows = math.ceil(num_frames / window_size)

        for i in range(num_windows):
            label_dict[gloss] += 1
            dataset.append((video_path, label, num_frames, i * window_size))

    print(f"The labels and label count is {label_dict}")

    return dataset


def build_balanced_dataset(cngt_video_paths: list, sb_video_paths: list, sb_vocab: dict, mode: str,
                           class_encodings: dict, window_size: int, split: str, random_seed: int) -> list:
    gloss_to_id = sb_vocab['gloss_to_id']

    class_cnt_dict = {}
    class_paths_dict = {}

    # count the total number of frames that each class will have, separate the video paths into classes
    for cngt_video_path in cngt_video_paths:

        metadata = load_gzip(cngt_video_path[:cngt_video_path.rfind(".m")] + ".gzip")
        num_frames = metadata.get("num_frames")

        gloss_id = int(cngt_video_path.split("_")[-1][:-4])
        gloss = list(gloss_to_id.keys())[list(gloss_to_id.values()).index(gloss_id)]

        # count number of frames taking into account window expansion
        if gloss not in class_cnt_dict.keys():
            class_cnt_dict[gloss] = 0

        num_windows = math.ceil(num_frames / window_size)
        class_cnt_dict[gloss] += window_size * num_windows

        # separate video paths based on class. Store number of frames to avoid reading metadata again later on
        if gloss not in class_paths_dict.keys():
            class_paths_dict[gloss] = [(cngt_video_path, num_frames)]
        else:
            class_paths_dict[gloss].append((cngt_video_path, num_frames))

    total_num_frames = sum(class_cnt_dict.values())
    train_size = int(total_num_frames * (4 / 6))
    val_test_size = int(total_num_frames * (1 / 6))

    # get the frames' ratio for each class to respect it when creating the splits
    ratio_dict = {key: class_cnt_dict[key] / total_num_frames for key in class_cnt_dict.keys()}

    # calculate the actual size of the splits for each class
    train_size_dict = {key: int(train_size * ratio_dict[key]) for key in ratio_dict.keys()}
    val_test_size_dict = {key: int(val_test_size * ratio_dict[key]) for key in ratio_dict.keys()}

    # setting a seed allows that the splits are always the same, regardless of when the data is loaded
    random.seed(random_seed)
    # randomize the order of the paths
    for key in class_paths_dict.keys():
        random.shuffle(class_paths_dict[key])

    # find out which video paths go in which split in order to load them later on. Paths are assigned until the previously calculated
    # size of each split is filled, based on the fact that only multiples of the window will be loaded during training
    cngt_split_dict = {}
    for fold in {'train', 'val', 'test'}:
        fold_paths = []
        for gloss in class_paths_dict.keys():
            max_windows = train_size_dict[gloss] // window_size if fold == 'train' else val_test_size_dict[gloss] // window_size
            windows_filled = 0
            while windows_filled < max_windows:
                if len(class_paths_dict[gloss]) == 0:
                    break
                path, num_frames = class_paths_dict[gloss].pop()
                fold_paths.append(path)
                num_windows = math.ceil(num_frames / window_size)
                windows_filled += num_windows
        cngt_split_dict[fold] = fold_paths

    # signbank videos are so few that can randomly assigned
    sb_idx_train_val = int(len(sb_video_paths) * (4 / 6))
    sb_idx_val_test = int(len(sb_video_paths) * (5 / 6))
    sb_folds = {'train': sb_video_paths[:sb_idx_train_val],
                'val': sb_video_paths[sb_idx_train_val:sb_idx_val_test],
                'test': sb_video_paths[sb_idx_val_test:]}

    # put together all video paths that correspond to the split being loaded
    all_video_paths = cngt_split_dict[split]
    all_video_paths.extend(sb_folds[split])

    num_classes = len(class_encodings)
    dataset = []
    label_dict = {}

    for video_path in tqdm(all_video_paths):
        metadata = load_gzip(video_path[:video_path.rfind(".m")] + ".gzip")
        num_frames = metadata.get("num_frames")
        if video_path.endswith(".mpg") or video_path.endswith(".mov"):  # cngt video
            gloss_id = int(video_path.split("_")[-1][:-4])
        else:
            gloss_id = int(video_path.split("-")[-1][:-4])

        # if mode == 'flow':
        #     num_frames = num_frames // 2

        label = np.zeros(num_classes, np.float32)
        label_idx = class_encodings[gloss_id]
        gloss = list(gloss_to_id.keys())[list(gloss_to_id.values()).index(gloss_id)]

        if gloss not in label_dict.keys():
            label_dict[gloss] = 0

        label[label_idx] = 1

        # if testing, we just keep 1 datapoint per test video
        if split == 'test':
            label_dict[gloss] += 1
            dataset.append((video_path, label, num_frames, 0))
        else:
            num_windows = math.ceil(num_frames / window_size)
            for i in range(num_windows):
                label_dict[gloss] += 1
                dataset.append((video_path, label, num_frames, i * window_size))

    print(f"The labels and label count is {label_dict}")

    return dataset


def build_random_dataset(cngt_video_paths: list, sb_video_paths: list, sb_vocab: dict, mode: str,
                         class_encodings: dict, window_size: int, split: str, random_seed: int) -> list:
    random.seed(random_seed)
    random.shuffle(cngt_video_paths)
    random.shuffle(sb_video_paths)

    # split dataset in 4:1:1 ratio
    cngt_train_val_idx = int(len(cngt_video_paths) * (4 / 6))
    cngt_val_test_idx = int(len(cngt_video_paths) * (5 / 6))

    cngt_folds = {'train': cngt_video_paths[:cngt_train_val_idx],
                  'val': cngt_video_paths[cngt_train_val_idx:cngt_val_test_idx],
                  'test': cngt_video_paths[cngt_val_test_idx:]}

    sb_train_val_idx = int(len(sb_video_paths) * (4 / 6))
    sb_val_test_idx = int(len(sb_video_paths) * (5 / 6))

    sb_folds = {'train': sb_video_paths[:sb_train_val_idx],
                'val': sb_video_paths[sb_train_val_idx:sb_val_test_idx],
                'test': sb_video_paths[sb_val_test_idx:]}

    # put together all video paths that correspond to the split being loaded
    all_video_paths = cngt_folds[split]
    all_video_paths.extend(sb_folds[split])

    num_classes = len(class_encodings)
    label_dict = {}
    dataset = []

    for video_path in tqdm(all_video_paths):
        metadata = load_gzip(video_path[:video_path.rfind(".m")] + ".gzip")
        num_frames = metadata.get("num_frames")
        if video_path.endswith(".mpg") or video_path.endswith(".mov"):  # cngt video
            gloss_id = int(video_path.split("_")[-1][:-4])
        else:
            gloss_id = int(video_path.split("-")[-1][:-4])

        # if mode == 'flow':
        #     num_frames = num_frames // 2

        label = np.zeros(num_classes, np.float32)
        label_idx = class_encodings[gloss_id]
        gloss = sb_vocab["id_to_gloss"][gloss_id]

        if gloss not in label_dict.keys():
            label_dict[gloss] = 0

        label[label_idx] = 1

        num_windows = math.ceil(num_frames / window_size)

        for i in range(num_windows):
            label_dict[gloss] += 1
            dataset.append((video_path, label, num_frames, i * window_size))

    print(f"The labels and label count is {label_dict}")

    return dataset


# def build_dataset_from_gzip(cngt_zip: str, sb_zip: str, sb_vocab_path: str, class_encodings: dict,
#                             window_size: int, diagonal_videos_path: str):
#     # process zip files first
#     if not os.path.isdir(cngt_zip[:-4]):
#         cngt_extracted_root = extract_zip(cngt_zip)
#     else:
#         cngt_extracted_root = cngt_zip[:-4]
#         # print(f"{cngt_extracted_root} already exists, no need to extract")
#
#     if not os.path.isdir(sb_zip[:-4]):
#         sb_extracted_root = extract_zip(sb_zip)
#     else:
#         sb_extracted_root = sb_zip[:-4]
#         # print(f"{sb_extracted_root} already exists, no need to extract")
#
#     num_classes = len(class_encodings)
#     dataset = []
#
#     sb_vocab = load_gzip(sb_vocab_path)
#     id_to_gloss = sb_vocab['id_to_gloss']
#
#     diagonal_videos = list(set(load_gzip(diagonal_videos_path)))
#     all_video_paths = []
#
#     for video in diagonal_videos:
#         if video.endswith(".mpg"):  # cngt video
#             all_video_paths.append(os.path.join(cngt_extracted_root, video))
#         else:
#             all_video_paths.append(os.path.join(sb_extracted_root, video))
#
#     label_dict = {}
#
#     for video_path in tqdm(all_video_paths):
#         metadata = load_gzip(video_path[:video_path.rfind(".m")] + ".gzip")
#         num_frames = metadata.get("num_frames")
#         if video_path.endswith(".mpg"):  # cngt video
#             gloss_id = int(video_path.split("_")[-1][:-4])
#         else:
#             gloss_id = int(video_path.split("-")[-1][:-4])
#
#         label = np.zeros(num_classes, np.float32)
#         label_idx = class_encodings[gloss_id]
#         gloss = id_to_gloss[gloss_id]
#
#         if gloss not in label_dict.keys():
#             label_dict[gloss] = 0
#
#         label[label_idx] = 1
#
#         num_windows = math.ceil(num_frames / window_size)
#
#         for i in range(num_windows):
#             label_dict[gloss] += 1
#             dataset.append((video_path, label, num_frames, i * window_size))
#
#     print(f"The labels and label count is {label_dict}")
#
#     return dataset


def build_dataset(loading_mode: str, cngt_root: str, sb_root: str, sb_vocab_path: str, mode: str,
                  class_encodings: dict, window_size: int, split: str, clips_per_class: int, random_seed: int) -> list:
    assert split in {"train", "val", "test"}, "The splits can only have value 'train', 'val', and 'test'."

    classes = list(class_encodings.keys())

    # check for zip files
    if not os.path.isdir(cngt_root) and os.path.exists(cngt_root + '.zip'):
        cngt_root = extract_zip(cngt_root + '.zip')

    if not os.path.isdir(sb_root) and os.path.exists(sb_root + '.zip'):
        sb_root = extract_zip(sb_root + '.zip')

    cngt_videos = [file for file in os.listdir(cngt_root) if file.endswith('.mpg') or file.endswith('.mov')]
    sb_videos = [file for file in os.listdir(sb_root) if file.endswith('.mp4')]

    # use signbank vocab to be able to get the glosses from their IDs
    sb_vocab = load_gzip(sb_vocab_path)

    # if a number of clips is specified, maybe a previous filtering of the videos
    if clips_per_class != -1 and clips_per_class > 0:

        random.seed(random_seed)
        subsample_cngt_videos = []
        subsample_sb_videos = []
        for gloss_class in classes:
            cngt_videos_single_class = [os.path.join(cngt_root, video) for video in cngt_videos if int(video.split("_")[-1][:-4]) == gloss_class]
            sb_videos_single_class = [os.path.join(sb_root, video) for video in sb_videos if int(video.split("-")[-1][:-4]) == gloss_class]

            random.shuffle(cngt_videos_single_class)
            random.shuffle(sb_videos_single_class)

            subsample_sb_videos.extend(sb_videos_single_class)
            subsample_cngt_videos.extend(cngt_videos_single_class[:clips_per_class - len(sb_videos_single_class)])

            cngt_video_paths = subsample_cngt_videos
            sb_video_paths = subsample_sb_videos
    else:
        cngt_video_paths = [os.path.join(cngt_root, video) for video in cngt_videos if int(video.split("_")[-1][:-4]) in classes]
        sb_video_paths = [os.path.join(sb_root, video) for video in sb_videos if int(video.split("-")[-1][:-4]) in classes]

    if loading_mode == "random":
        dataset = build_random_dataset(cngt_video_paths, sb_video_paths, sb_vocab, mode, class_encodings, window_size, split, random_seed)
    elif loading_mode == "balanced":
        dataset = build_balanced_dataset(cngt_video_paths, sb_video_paths, sb_vocab, mode, class_encodings, window_size, split, random_seed)
    elif loading_mode == "stratified":
        dataset = build_stratified_dataset(cngt_video_paths, sb_video_paths, sb_vocab, mode, class_encodings, window_size, split, random_seed)

    return dataset


def get_class_encodings(cngt_root, sb_root, filter_num=None, specific_gloss_ids=[]):
    # check for zip files
    if not os.path.isdir(cngt_root) and os.path.exists(cngt_root + '.zip'):
        cngt_root = extract_zip(cngt_root + '.zip')

    if not os.path.isdir(sb_root) and os.path.exists(sb_root + '.zip'):
        sb_root = extract_zip(sb_root + '.zip')

    cngt_gloss_ids = [int(video.split("_")[-1][:-4]) for video in os.listdir(cngt_root) if video.endswith('.mpg') or video.endswith('.mov')]
    sb_gloss_ids = {}

    if filter_num is None:  # default case
        if len(specific_gloss_ids) > 0:  # if some glosses are specified
            assert type(specific_gloss_ids) == list, "The variable 'specific_glosses' must be a list"
            assert type(specific_gloss_ids[0]) == int, "The variable 'specific_glosses' must be a list of integers"
            cngt_gloss_ids = specific_gloss_ids
        else:
            sb_gloss_ids = [int(video.split("-")[-1][:-4]) for video in os.listdir(sb_root) if video.endswith('.mp4')]
    else:
        assert type(filter_num) == int, "The variable 'filter_num' must be an integer"
        _, top_cngt_ids = filter_top_glosses(cngt_gloss_ids, filter_num)
        cngt_gloss_ids = top_cngt_ids

    classes = list(set(cngt_gloss_ids).union(set(sb_gloss_ids)))

    class_to_idx = {}
    for i in range(len(classes)):
        class_to_idx[classes[i]] = i

    return class_to_idx


class I3Dataset(data_utl.Dataset):

    def __init__(self, loading_mode, cngt_root, sb_root, sb_vocab_path, mode, split, window_size=64, transforms=None,
                 filter_num=None, specific_gloss_ids=None, clips_per_class=-1, random_seed=42):

        assert loading_mode in {'random', 'balanced', 'stratified'}, "The 'loading_mode' argument must have values 'random', 'balanced', 'stratified'"
        assert mode in {'rgb', 'flow'}, "The 'mode' argument must have values 'rgb' or 'flow'"

        self.mode = mode
        self.class_encodings = get_class_encodings(cngt_root, sb_root, filter_num, specific_gloss_ids)
        self.window_size = window_size
        self.transforms = transforms
        self.data = build_dataset(loading_mode, cngt_root, sb_root, sb_vocab_path, mode, self.class_encodings, window_size, split, clips_per_class, random_seed)

    def __getitem__(self, index):
        video_path, label, num_frames, start_frame = self.data[index]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(video_path, start_frame, self.window_size)
        # FLOW NOT IMPLEMENTED
        # else:
        #     imgs = load_flow_frames(self.root, vid, start_frame)

        # pytorch transforms take input tensor of shape [C, T, H, W]
        imgs = imgs.permute([3, 0, 1, 2])

        if self.transforms:
            imgs = self.transforms(imgs)

        return imgs, torch.from_numpy(label), video_path

    def __len__(self):
        return len(self.data)