import os
import sys

import pandas as pd

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

import argparse
import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import itertools

from src.utils.i3d_data import I3Dataset
from src.utils.helpers import load_config, make_dir
from src.utils.pytorch_i3d import InceptionI3d
from src.utils.i3d_dimensions_conv import InceptionI3d as InceptionDimsConv
from src.utils.util import load_gzip, save_gzip
from scipy.spatial import distance
from pathlib import Path
from torchvision import transforms
import seaborn as sns
from src.utils.geomle import geomle
from src.utils.mle import mle as mle_normal
from src.utils.corrected_mle import mle, mle_inverse_singlek, mle_inverse_singlek_loop
from src.utils.twonn import twonn_dimension as twonn
from skdim import id
import math

def stress(X_pred, X):
    # distance of every point (row) to the rest of points in matrix
    orig_dist = distance.pdist(X, 'euclidean')
    pred_dist = distance.pdist(X_pred, 'euclidean')
    # stress formula from http://analytictech.com/networks/mds.htm
    return np.sqrt(sum((pred_dist - orig_dist) ** 2) / sum(orig_dist ** 2))

def dim_red(specific_glosses: list, config: dict, fig_output_root: str):

    pca_config = config.get("pca")
    data_config = config.get("data")

    # pca parameters
    fold = pca_config.get("fold")
    assert fold in {"train", "test", "val"}, f"Parameter 'fold' is {fold} but should be either 'train' 'val' or 'test'"

    batch_size = pca_config.get("batch_size")
    run_name = pca_config.get("run_name")
    run_batch_size = pca_config.get("run_batch_size")
    run_lr = pca_config.get("run_lr")
    run_optimizer = pca_config.get("run_optimizer")
    run_epochs = pca_config.get("run_epochs")
    model_root = pca_config.get("model_root")
    use_cuda = pca_config.get("use_cuda")
    random_seed = pca_config.get("random_seed")

    # data configs
    clips_per_class = data_config.get("clips_per_class")
    root = data_config.get("root")
    cngt_clips_folder = data_config.get("cngt_clips_folder")
    signbank_folder = data_config.get("signbank_folder")
    sb_vocab_file = data_config.get("sb_vocab_file")
    window_size = data_config.get("window_size")
    loading_mode = data_config.get("loading_mode")
    input_size = data_config.get("input_size")

    cngt_root = os.path.join(root, cngt_clips_folder)
    sb_root = os.path.join(root, signbank_folder)
    sb_vocab_path = os.path.join(root, sb_vocab_file)

    # get directory and filename for the checkpoints
    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    run_dir = f"{run_name}_{glosses_string}_{run_epochs}_{run_batch_size}_{run_lr}_{run_optimizer}"

    ckpt_files = [file for file in os.listdir(os.path.join(model_root, run_dir)) if file.endswith(".pt")]
    # take the last save checkpoint, which contains the minimum val loss
    ckpt_filename = ckpt_files[-1]
    # ckpt_filename = f"i3d_{str(ckpt_epoch).zfill(len(str(run_epochs)))}.pt"

    num_top_glosses = None

    sb_vocab = load_gzip(sb_vocab_path)
    gloss_to_id = sb_vocab['gloss_to_id']

    specific_gloss_ids = [gloss_to_id[gloss] for gloss in specific_glosses]

    cropped_input_size = input_size * 0.875

    test_transforms = transforms.Compose([transforms.CenterCrop(cropped_input_size)])

    print(f"Loading {fold} split...")
    dataset = I3Dataset(loading_mode=loading_mode,
                        cngt_root=cngt_root,
                        sb_root=sb_root,
                        sb_vocab_path=sb_vocab_path,
                        mode="rgb",
                        split=fold,
                        window_size=window_size,
                        transforms=test_transforms,
                        filter_num=num_top_glosses,
                        specific_gloss_ids=specific_gloss_ids,
                        clips_per_class=clips_per_class,
                        random_seed=random_seed)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    i3d = InceptionI3d(num_classes=len(dataset.class_encodings),
                       in_channels=3,
                       window_size=window_size,
                       input_size=cropped_input_size)

    if use_cuda:
        i3d.load_state_dict(torch.load(os.path.join(model_root, run_dir, ckpt_filename)))
    else:
        i3d.load_state_dict(torch.load(os.path.join(model_root, run_dir, ckpt_filename), map_location=torch.device('cpu')))

    if use_cuda:
        i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode

    # we initialize with shape (1, 1024) since we don't know the final size of the matrix
    X_features = torch.zeros((1, 1024))
    X = torch.zeros((1, 1024))
    Y = np.zeros(1)

    print(f"Running datapoints through model...")
    with torch.no_grad():  # this deactivates gradient calculations, reducing memory consumption by A LOT
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                # get the inputs
                inputs, labels, _ = data
                if use_cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                # get the features of the penultimate layer
                features = i3d.extract_features(inputs)
                features = torch.squeeze(features, -1)

                # if X_features is 0 (empty)
                if X_features.sum() == 0:
                    X_features = features.squeeze()
                else:
                    # if the last batch has only 1 video, the squeeze function removes an extra dimension and cannot be concatenated
                    if len(features.squeeze().size()) == 1:
                        X_features = torch.cat((X_features, torch.unsqueeze(features.squeeze(), 0)), dim=0)
                    else:
                        X_features = torch.cat((X_features, features.squeeze()), dim=0)

    X_features = X_features.detach().cpu()

    mle_id, _ = mle_normal(pd.DataFrame(X_features), k1=5, k2=20, average=True)
    # mle_inv_id = mle_inverse_singlek_loop(X_features, k1=5, k2=20, k_step=5, average=True)
    # mle_corr_id = mle(X_features, average=True)
    # mle_inv_id = mle_inverse_singlek(X_features, k1=20)

    # mle_id = MLE().fit_transform(X_features)
    danco_id = id.DANCo().fit_transform(X_features)
    mindmli_id = id.MiND_ML(ver='MLi').fit_transform(X_features)
    mindmlk_id = id.MiND_ML(ver='MLk').fit_transform(X_features)
    lpca_id = id.lPCA().fit_transform(X_features)
    mom_id = id.MOM().fit_transform(X_features)
    tle_id = id.TLE().fit_transform(X_features)

    print(f"Intrinsic dimensions for binary classification of {specific_glosses}:"
          f"\nMLE={mle_id}"
          f"\nDANCo={danco_id}"
          f"\nlPCA={lpca_id}"
          f"\nMiND_MLk={mindmlk_id}"
          f"\nMiND_MLi={mindmli_id}"
          f"\nMOM={mom_id}"
          f"\nTLE={tle_id}")

    return

def main(params):
    config_path = params.config_path
    fig_output_root = params.fig_output_root

    config = load_config(config_path)
    pca_config = config.get("pca")

    reference_sign = pca_config.get("reference_sign")
    signs = pca_config.get("signs")
    # ckpt_epoch_list = pca_config.get("ckpt_epoch_list")

    assert type(signs) == list, "The variable 'train_signs' must be a list."
    # assert type(ckpt_epoch_list) == list, "The variable 'ckpt_epoch_list' must be a list."
    # assert len(signs) == len(ckpt_epoch_list), "Every sign pair needs to have a corresponding checkpoint."

    # intr_dims = []
    for i, sign in enumerate(signs):
        dim_red(specific_glosses=[reference_sign, sign], config=config, fig_output_root=fig_output_root)

    # print(f"Intrinsic dimensions between reference sign {reference_sign} and {signs}:\n{intr_dims}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
    )

    params, _ = parser.parse_known_args()
    main(params)
