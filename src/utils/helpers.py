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

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg