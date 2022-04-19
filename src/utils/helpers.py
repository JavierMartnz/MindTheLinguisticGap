import yaml
import torch
import numpy as np
import random
import _pickle as cPickle
import gzip
import torch
from torch import nn, Tensor
import os
import os.path
import shutil

import logging
from logging import Logger
import errno
import glob
from typing import Optional
import tarfile
import json
import re

class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """

def make_model_dir(model_dir: str, overwrite=False):
    init=False
    if os.path.isdir(model_dir):
        if not overwrite:
            # raise FileExistsError("Model directory %s exsists and overwriting is disabled." % model_dir)
            init=True
            return model_dir, init
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir, init

def make_dir(dir: str) -> str:
    if not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def make_logger(log_file: str = None) -> Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logging.getLogger("").addHandler(sh)
    logger.info("Hello! This is BERT-skeleton")
    return logger


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object

def load_openpose_tar_xz(filename):
    keypoints = {}
    tar = tarfile.open(filename)
    for member in tar.getmembers():
        if member.isfile():
            filename = os.path.abspath(member.name)
            # print(filename)
            num_frame = re.findall('\d{12,}', filename)
            num_frame = num_frame[0].lstrip('0')  # tj : the first result
            if num_frame == '':
                num_frame = 0
            num_frame = int(num_frame)
            f = tar.extractfile(member)
            keypoints[num_frame] = json.loads(f.read())
    return keypoints

def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime) # tj : getctime : get file created time

    # check existence
    if latest_checkpoint is None:
        raise FileNotFoundError("No checkpoint found in directory {}."
                                .format(ckpt_dir))
    return latest_checkpoint

def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))

def overwrite_file(filename, info) -> None:
    with open(filename, 'w') as opened_file:
        opened_file.write( info )

def append_file(filename, info) -> None:
    with open(filename, 'a') as opened_file:
        opened_file.write(info)


def embed_text(input_video_name, output_video_name, text):
    embed_text_command = 'ffmpeg -y -i ' +  input_video_name + ' -vf "[in]drawtext= fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:' +\
    'text=' + text + ': fontcolor=white: fontsize=36: box=1: boxcolor=black@0.5: boxborderw=5: x=(w-text_w)/2:y=7*(h-text_h)/8[out]" -pix_fmt yuv420p ' +\
    output_video_name
    print(embed_text_command)
    os.system(embed_text_command)

def embed_2text(input_video_name, output_video_name, text1, text2, linebreak):
    embed_text_command = 'ffmpeg -y -i ' + input_video_name + ' -vf "[in]drawtext= fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:' + \
                         'text=' + text1 + ': fontcolor=white: fontsize=36: box=1: boxcolor=black@0.5: boxborderw=5: x=(w-text_w)/2:y=7*(h-text_h)/8, ' +  \
                         'drawtext= fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:' + \
                         'text=' + text2 + ': fontcolor=white: fontsize=36: box=1: boxcolor=black@0.5: boxborderw=5: x=(w-text_w)/2:y=7*(h-text_h)/8+' + str(linebreak) + '[out]" -pix_fmt yuv420p ' + \
                         output_video_name
    print(embed_text_command)
    os.system(embed_text_command)

    #fontfile=/vol/research/SignPose/tj/Workspace/Font/Ubuntu-B.ttf:



def f1_score_numpy(TP, TN, FP, FN):
    acc = (TP) / (TN + TP + FN + FP)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    F1 = 2 * precision * recall / ( precision + recall + 1e-6)
    return acc, F1, precision, recall

import time

class Block:
    """A minimal inline codeblock timer"""
    def __init__(
            self,
            name: str,
    ):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0