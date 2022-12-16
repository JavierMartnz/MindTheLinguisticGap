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
from src.utils.util import load_gzip
from scipy.spatial import distance
from MulticoreTSNE import MulticoreTSNE as TSNE


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, root_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if root_path:
        plt.savefig(os.path.join(root_path, "confusion_matrix"))
    # plt.show()

def stress(X_pred, X):
    # distance of every point (row) to the rest of points in matrix
    orig_dist = distance.pdist(X, 'euclidean')
    pred_dist = distance.pdist(X_pred, 'euclidean')
    # stress formula from http://analytictech.com/networks/mds.htm
    return np.sqrt(sum((pred_dist - orig_dist)**2)/sum(orig_dist**2))

def tsne(cfg_path, log_filename, mode="rgb"):
    cfg = load_config(cfg_path)
    test_cfg = cfg.get("test")
    data_cfg = cfg.get("data")

    # test parameters
    model_dir = test_cfg.get("model_dir")
    pred_dir = test_cfg.get("pred_dir")
    fold = test_cfg.get("fold")
    assert fold in {"train", "test",
                    "val"}, f"Please, make sure the parameter 'fold' in {cfg_path} is either 'train' 'val' or 'test'"
    run_name = test_cfg.get("run_name")
    run_batch_size = test_cfg.get("run_batch_size")
    optimizer = test_cfg.get("optimizer").upper()
    learning_rate = test_cfg.get("lr")
    num_epochs = test_cfg.get("epochs")
    ckpt_epoch = test_cfg.get("ckpt_epoch")
    ckpt_step = test_cfg.get("ckpt_step")
    batch_size = test_cfg.get("batch_size")
    use_cuda = test_cfg.get("use_cuda")
    specific_glosses = test_cfg.get("specific_glosses")
    extra_conv = test_cfg.get("extra_conv")

    # data configs
    cngt_zip = data_cfg.get("cngt_clips_path")
    sb_zip = data_cfg.get("signbank_path")
    window_size = data_cfg.get("window_size")
    cngt_vocab_path = data_cfg.get("cngt_vocab_path")
    sb_vocab_path = data_cfg.get("sb_vocab_path")
    loading_mode = data_cfg.get("data_loading")
    use_diag_videos = data_cfg.get("use_diag_videos")
    if use_diag_videos:
        diag_videos_path = data_cfg.get("diagonal_videos_path")
    else:
        diag_videos_path = None
    final_pooling_size = data_cfg.get("final_pooling_size")

    # get directory and filename for the checkpoints
    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    run_dir = f"{run_name}_{glosses_string}_{num_epochs}_{run_batch_size}_{learning_rate}_{optimizer}"
    # run_dir = f"b{run_batch_size}_{optimizer}_lr{learning_rate}_ep{num_epochs}_{run_name}"
    ckpt_filename = f"i3d_{str(ckpt_epoch).zfill(len(str(num_epochs)))}_{ckpt_step}.pt"
    ckpt_folder = ckpt_filename.split('.')[0]

    num_top_glosses = None

    # get glosses from the class encodings
    cngt_vocab = load_gzip(cngt_vocab_path)
    sb_vocab = load_gzip(sb_vocab_path)
    # join cngt and sb vocabularies (gloss to id dictionary)
    sb_vocab.update(cngt_vocab)
    gloss_to_id = sb_vocab['gloss_to_id']

    specific_gloss_ids = [gloss_to_id[gloss] for gloss in specific_glosses]

    print(f"Loading {fold} split...")
    dataset = I3Dataset(loading_mode, cngt_zip, sb_zip, cngt_vocab_path, sb_vocab_path, mode, fold, window_size,
                        transforms=None,
                        filter_num=num_top_glosses, specific_gloss_ids=specific_gloss_ids,
                        diagonal_videos_path=diag_videos_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    # load model and specified checkpoint
    if extra_conv:
        i3d = InceptionDimsConv(num_classes=len(dataset.class_encodings), in_channels=3, window_size=16,
                            conv_output_dims=final_pooling_size)
        i3d.add_dim_conv()
        i3d.replace_logits(num_classes=len(dataset.class_encodings))
    else:
        i3d = InceptionI3d(num_classes=len(dataset.class_encodings), in_channels=3, window_size=16)


    if use_cuda:
        i3d.load_state_dict(torch.load(os.path.join(model_dir, run_dir, ckpt_filename)))
    else:
        i3d.load_state_dict(
            torch.load(os.path.join(model_dir, run_dir, ckpt_filename), map_location=torch.device('cpu')))

    if use_cuda:
        i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    ckpt_folder = ckpt_filename.split('.')[0]
    pred_path = os.path.join(pred_dir, run_dir, ckpt_folder, fold)
    make_dir(pred_path)

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
                    X = preds.squeeze()
                    Y = y_true
                else:
                    # if the last batch has only 1 video, the squeeze function removes an extra dimension and cannot be concatenated
                    if len(features.squeeze().size()) == 1:
                        X_features = torch.cat((X_features, torch.unsqueeze(features.squeeze(), 0)), dim=0)
                        X = torch.cat((X, torch.unsqueeze(preds.squeeze(), 0)), dim=0)
                    else:
                        X_features = torch.cat((X_features, features.squeeze()), dim=0)
                        X = torch.cat((X, preds.squeeze()), dim=0)

                    Y = np.append(Y, y_true)

    print("Running TSNE...")
    GLOSS1 = Y == 0
    GLOSS2 = Y == 1

    X = X.detach().cpu()
    X_features = X_features.detach().cpu()

    n_components = 16

    X_tsne = TSNE(n_components=n_components, perplexity=200).fit_transform(X)

    print(f"The stress for {n_components} dimensions is {round(stress(X_tsne, X_features), 4)}")

def main(params):
    config_path = params.config_path
    log_filename = params.log_filename
    tsne(config_path, log_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
    )

    parser.add_argument(
        "--log_filename",
        type=str,
        default="test_metrics.txt"
    )

    params, _ = parser.parse_known_args()
    main(params)
