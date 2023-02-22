import os
import sys

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
from sklearn.decomposition import PCA
from pathlib import Path

def stress(X_pred, X):
    # distance of every point (row) to the rest of points in matrix
    orig_dist = distance.pdist(X, 'euclidean')
    pred_dist = distance.pdist(X_pred, 'euclidean')
    # stress formula from http://analytictech.com/networks/mds.htm
    return np.sqrt(sum((pred_dist - orig_dist)**2)/sum(orig_dist**2))

def main(params):
    config_path = params.config_path
    fig_output_root = params.fig_output_root

    cfg = load_config(config_path)
    test_cfg = cfg.get("test")
    data_cfg = cfg.get("data")

    # test parameters
    model_dir = test_cfg.get("model_dir")
    fold = test_cfg.get("fold")
    assert fold in {"train", "test",
                    "val"}, f"Please, make sure the parameter 'fold' in {config_path} is either 'train' 'val' or 'test'"
    run_name = test_cfg.get("run_name")
    run_batch_size = test_cfg.get("run_batch_size")
    optimizer = test_cfg.get("optimizer").upper()
    learning_rate = test_cfg.get("lr")
    num_epochs = test_cfg.get("epochs")
    ckpt_epoch = test_cfg.get("ckpt_epoch")
    batch_size = test_cfg.get("batch_size")
    use_cuda = test_cfg.get("use_cuda")
    specific_glosses = test_cfg.get("specific_glosses")

    # data configs
    cngt_zip = data_cfg.get("cngt_clips_path")
    sb_zip = data_cfg.get("signbank_path")
    window_size = data_cfg.get("window_size")
    sb_vocab_path = data_cfg.get("sb_vocab_path")
    loading_mode = data_cfg.get("data_loading")
    input_size = data_cfg.get("input_size")

    # get directory and filename for the checkpoints
    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    run_dir = f"{run_name}_{glosses_string}_{num_epochs}_{run_batch_size}_{learning_rate}_{optimizer}"
    ckpt_filename = f"i3d_{str(ckpt_epoch).zfill(len(str(num_epochs)))}.pt"

    num_top_glosses = None

    sb_vocab = load_gzip(sb_vocab_path)
    gloss_to_id = sb_vocab['gloss_to_id']

    specific_gloss_ids = [gloss_to_id[gloss] for gloss in specific_glosses]

    print(f"Loading {fold} split...")
    dataset = I3Dataset(loading_mode=loading_mode,
                        cngt_zip=cngt_zip,
                        sb_zip=sb_zip,
                        sb_vocab_path=sb_vocab_path,
                        mode="rgb",
                        split=fold,
                        window_size=window_size,
                        transforms=None,
                        filter_num=num_top_glosses,
                        specific_gloss_ids=specific_gloss_ids)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    i3d = InceptionI3d(num_classes=len(dataset.class_encodings), in_channels=3, window_size=window_size,
                       input_size=input_size)

    if use_cuda:
        i3d.load_state_dict(torch.load(os.path.join(model_dir, run_dir, ckpt_filename)))
    else:
        i3d.load_state_dict(
            torch.load(os.path.join(model_dir, run_dir, ckpt_filename), map_location=torch.device('cpu')))

    if use_cuda:
        i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode

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
                preds = i3d(inputs)

                features = torch.squeeze(features, -1)
                preds = torch.squeeze(preds, -1)

                y_true = np.argmax(labels.detach().cpu().numpy(), axis=1)

                # if X is empty
                if X_features.sum() == 0:
                    X_features = features.squeeze()
                    X_pred = preds.squeeze()
                    Y = y_true
                else:
                    # if the last batch has only 1 video, the squeeze function removes an extra dimension and cannot be concatenated
                    if len(features.squeeze().size()) == 1:
                        X_features = torch.cat((X_features, torch.unsqueeze(features.squeeze(), 0)), dim=0)
                        X_pred = torch.cat((X_pred, torch.unsqueeze(preds.squeeze(), 0)), dim=0)
                    else:
                        X_features = torch.cat((X_features, features.squeeze()), dim=0)
                        X_pred = torch.cat((X_pred, preds.squeeze()), dim=0)

                    Y = np.append(Y, y_true)

    X_features = X_features.detach().cpu()

    n_components = 2 ** np.arange(1, 11)[::-1]

    pca_stress = []
    print("Running PCA...")
    for nc in n_components:
        try:
            X_pca = PCA(n_components=nc).fit_transform(X_features)
            pca_stress.append(stress(X_pca, X_features))
            print(f"The stress from 1024 to {nc} dimensions is {round(stress(X_pca, X_features), 4)}")
        except Exception as e:
            print(e)

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    plt.plot(n_components, pca_stress)
    plt.xticks(n_components)
    plt.xlabel("Number of dimensions")
    plt.ylabel("Stress")
    plt.tight_layout()

    os.makedirs(fig_output_root, exist_ok=True)
    plt.savefig(os.path.join(fig_output_root, run_dir + '_pcastress.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
    )

    parser.add_argument(
        "--fig_output_root",
        type=str,
        default="D:/Thesis/graphs"
    )

    params, _ = parser.parse_known_args()
    main(params)
