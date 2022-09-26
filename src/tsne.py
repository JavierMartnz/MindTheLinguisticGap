import math
import os
import sys

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef as MCC
import matplotlib.pyplot as plt
import itertools
from torchvision.utils import save_image

from src.utils.i3d_data import I3Dataset
from src.utils.helpers import load_config, make_dir
from src.utils.pytorch_i3d import InceptionI3d
from src.utils.util import load_gzip
from sklearn.manifold import TSNE


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


def tsne(cfg_path, log_filename, mode="rgb"):
    cfg = load_config(cfg_path)
    test_cfg = cfg.get("test")
    data_cfg = cfg.get("data")

    # test parameters
    model_dir = test_cfg.get("model_dir")
    pred_dir = test_cfg.get("pred_dir")
    fold = test_cfg.get("fold")
    assert fold in {"train", "test", "val"}, f"Please, make sure the parameter 'fold' in {cfg_path} is either 'train' 'val' or 'test'"
    run_name = test_cfg.get("run_name")
    run_batch_size = test_cfg.get("run_batch_size")
    optimizer = test_cfg.get("optimizer").upper()
    learning_rate = test_cfg.get("lr")
    num_epochs = test_cfg.get("epochs")
    ckpt_epoch = test_cfg.get("ckpt_epoch")
    ckpt_step = test_cfg.get("ckpt_step")
    batch_size = test_cfg.get("batch_size")
    use_cuda = test_cfg.get("use_cuda")

    # data configs
    cngt_zip = data_cfg.get("cngt_clips_path")
    sb_zip = data_cfg.get("signbank_path")
    window_size = data_cfg.get("window_size")
    cngt_vocab_path = data_cfg.get("cngt_vocab_path")
    sb_vocab_path = data_cfg.get("sb_vocab_path")
    loading_mode = data_cfg.get("data_loading")

    # get directory and filename for the checkpoints
    run_dir = f"b{run_batch_size}_{optimizer}_lr{learning_rate}_ep{num_epochs}_{run_name}"
    ckpt_filename = f"i3d_{str(ckpt_epoch).zfill(len(str(num_epochs)))}_{ckpt_step}.pt"

    num_top_glosses = 2

    print(f"Loading {fold} split...")
    dataset = I3Dataset(loading_mode, cngt_zip, sb_zip, cngt_vocab_path, sb_vocab_path, mode, fold, window_size, transforms=None,
                        filter_num=num_top_glosses)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # # get glosses from the class encodings
    # cngt_vocab = load_gzip("D:/Thesis/datasets/cngt_vocab.gzip")
    # sb_vocab = load_gzip("D:/Thesis/datasets/signbank_vocab.gzip")
    # # join cngt and sb vocabularies (gloss to id dictionary)
    # sb_vocab.update(cngt_vocab)
    # gloss_to_id = sb_vocab['gloss_to_id']
    #
    # glosses = []
    #
    # for gloss_id in dataset.class_encodings.keys():
    #     glosses.append(list(gloss_to_id.keys())[list(gloss_to_id.values()).index(gloss_id)])
    #
    # print(f"Predicting for glosses {glosses} mapped as classes {list(dataset.class_encodings.values())}")

    # load model and specified checkpoint
    i3d = InceptionI3d(num_classes=len(dataset.class_encodings), in_channels=3, window_size=16)
    if use_cuda:
        i3d.load_state_dict(torch.load(os.path.join(model_dir, run_dir, ckpt_filename)))
    else:
        i3d.load_state_dict(torch.load(os.path.join(model_dir, run_dir, ckpt_filename), map_location=torch.device('cpu')))

    if use_cuda:
        i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    total_pred = []
    total_true = []

    X = torch.zeros((16, 1024))
    Y = np.zeros(16)

    print(f"Running datapoints through model...")
    with torch.no_grad():  # this deactivates gradient calculations, reducing memory consumption by A LOT
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                # get the inputs
                inputs, labels = data
                if use_cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                y_true = np.max(np.argmax(labels.detach().cpu().numpy(), axis=1), axis=1)

                # get the features of the penultimate layer
                features = i3d.extract_features(inputs)

                # if X is empty
                if X.sum() == 0:
                    X = features.squeeze()
                    Y = y_true
                else:
                    X = torch.cat((X, features.squeeze()), dim=0)
                    Y = np.append(Y, y_true)

    JA = Y == 0
    GEBAREN = Y == 1

    perplexities = [1, 2, 5, 10, 20, 30, 50, 100]
    X_embeds = []
    for perplexity in perplexities:
        X_embeds.append(TSNE(n_components=2, perplexity=perplexity, n_jobs=-1).fit_transform(X.detach().cpu()
        ))

    fig, axs = plt.subplots(nrows=math.ceil(len(perplexities)/3), ncols=3, figsize=(15, 12))
    fig.suptitle(f"{fold} {run_name} {ckpt_filename}")
    for i, ax in enumerate(axs.ravel()):
        if i >= len(perplexities):
            break
        ax.scatter(X_embeds[i][JA, 0], X_embeds[i][JA, 1], c='orange', label="JA")
        ax.scatter(X_embeds[i][GEBAREN, 0], X_embeds[i][GEBAREN, 1], c='blue', label="GEBAREN")
        ax.set_title(f"Perplexity = {perplexities[i]}")
        ax.legend()

    plt.show()


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
