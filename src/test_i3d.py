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
import matplotlib.pyplot as plt
import itertools

from src.utils.i3d_data import I3Dataset
from src.utils.helpers import load_config
from src.utils.pytorch_i3d import InceptionI3d
from src.utils.util import load_gzip

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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
    plt.show()

def test(cfg_path, mode="rgb"):
    cfg = load_config(cfg_path)
    test_cfg = cfg.get("test")

    # test parameters
    model_dir = test_cfg.get("model_dir")
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
    cngt_zip = cfg.get("data").get("cngt_clips_path")
    sb_zip = cfg.get("data").get("signbank_path")
    window_size = cfg.get("data").get("window_size")

    # get directory and filename for the checkpoints
    run_dir = f"b{run_batch_size}_{optimizer}_lr{learning_rate}_ep{num_epochs}_{run_name}"
    ckpt_filename = f"i3d_{ckpt_epoch}_{ckpt_step}.pt"

    num_top_glosses = 2

    print("Loading test split...")
    dataset = I3Dataset(cngt_zip, sb_zip, mode, 'test', window_size, transforms=None, filter_num=num_top_glosses)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # get glosses from the class encodings
    cngt_vocab = load_gzip("D:/Thesis/datasets/cngt_vocab.gzip")
    sb_vocab = load_gzip("D:/Thesis/datasets/signbank_vocab.gzip")
    # join cngt and sb vocabularies (gloss to id dictionary)
    sb_vocab.update(cngt_vocab)
    gloss_to_id = sb_vocab['gloss_to_id']

    glosses = []

    for gloss_id in dataset.class_encodings.keys():
        glosses.append(list(gloss_to_id.keys())[list(gloss_to_id.values()).index(gloss_id)])

    print(f"Predicting for glosses {glosses} mapped as classes {list(dataset.class_encodings.values())}")

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

    print("Predicting on test set...")
    with torch.no_grad():  # this deactivates gradient calculations, reducing memory consumption by A LOT
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                # get the inputs
                inputs, labels = data
                if use_cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                # forward pass of the inputs through the network
                per_frame_logits = i3d(inputs)

                # upsample output to input size
                per_frame_logits = F.interpolate(per_frame_logits, size=inputs.size(2), mode='linear')

                # get prediction
                y_pred = np.argmax(per_frame_logits.detach().cpu().numpy(), axis=1)
                y_true = np.argmax(labels.detach().cpu().numpy(), axis=1)

                if len(total_pred) == 0:
                    total_pred = y_pred.flatten()
                    total_true = y_true.flatten()
                else:
                    total_pred = np.append(total_pred, y_pred.flatten())
                    total_true = np.append(total_true, y_true.flatten())

    f1 = f1_score(total_true, total_pred, average='macro')
    acc = accuracy_score(total_true, total_pred)
    cm = confusion_matrix(total_true, total_pred)
    plot_confusion_matrix(cm, glosses)

    print(f"F1 = {f1:.4f}\tAcc = {acc:.4f}")

def main(params):
    config_path = params.config_path
    test(config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
    )

    params, _ = parser.parse_known_args()
    main(params)