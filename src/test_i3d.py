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


def test(cfg_path, log_filename, mode="rgb"):
    cfg = load_config(cfg_path)
    test_cfg = cfg.get("test")

    # test parametersf
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
    cngt_zip = cfg.get("data").get("cngt_clips_path")
    sb_zip = cfg.get("data").get("signbank_path")
    window_size = cfg.get("data").get("window_size")
    cngt_vocab_path = cfg.get("data").get("cngt_vocab_path")
    sb_vocab_path = cfg.get("data").get("sb_vocab_path")

    # get directory and filename for the checkpoints
    run_dir = f"b{run_batch_size}_{optimizer}_lr{learning_rate}_ep{num_epochs}_{run_name}"
    ckpt_filename = f"i3d_{str(ckpt_epoch).zfill(len(str(num_epochs)))}_{ckpt_step}.pt"

    pred_path = os.path.join(pred_dir, run_dir, fold, ckpt_filename.split('.')[0])
    make_dir(pred_path)
    make_dir(os.path.join(pred_path, "TP"))
    make_dir(os.path.join(pred_path, "TN"))
    make_dir(os.path.join(pred_path, "FP"))
    make_dir(os.path.join(pred_path, "FN"))

    num_top_glosses = 2

    print(f"Loading {fold} split...")
    dataset = I3Dataset(cngt_zip, sb_zip, cngt_vocab_path, sb_vocab_path, mode, fold, window_size, transforms=None,
                        filter_num=num_top_glosses)
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

    print(f"Predicting on {fold} set...")
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

                # save predicitions for later
                if len(total_pred) == 0:
                    total_pred = y_pred.flatten()
                    total_true = y_true.flatten()
                else:
                    total_pred = np.append(total_pred, y_pred.flatten())
                    total_true = np.append(total_true, y_true.flatten())

                # swap to [B, T, H, W, C] for storing
                images = inputs.permute([0, 2, 1, 3, 4])

                for batch_sample in range(images.size(0)):
                    for frame in range(images.size(1)):
                        pred = y_pred[batch_sample][frame]
                        label = y_true[batch_sample][frame]
                        filename = f"b{batch_sample}_f{frame}.png"

                        # THIS IS DONE FOR BINARY CLASSIFICATION, SHOULD BE CHANGED FOR MULTICLASS
                        # CLASS 0 IS POSITIVE, 1 IS NEGATIVE
                        if label == pred and label == 0:  # TP
                            save_image(images[batch_sample][frame], os.path.join(pred_path, "TP", filename))
                        elif label == pred and label == 1:  # TN
                            save_image(images[batch_sample][frame], os.path.join(pred_path, "TN", filename))
                        elif label != pred and label == 0:  # FN
                            save_image(images[batch_sample][frame], os.path.join(pred_path, "FN", filename))
                        elif label != pred and label == 1:  # FP
                            save_image(images[batch_sample][frame], os.path.join(pred_path, "FP", filename))

    f1 = f1_score(total_true, total_pred, average='macro')
    acc = accuracy_score(total_true, total_pred)
    mcc = MCC(total_true, total_pred)
    cm = confusion_matrix(total_true, total_pred)

    print(f"F1 = {f1:.4f}\tAcc = {acc:.4f}\tMCC = {mcc:.4f}")
    print(cm)

    logfile_path = os.path.join(pred_path, log_filename)

    if os.path.exists(logfile_path):
        os.remove(logfile_path)
    with open(logfile_path, 'w') as f:
        print(f"Predicting for glosses {glosses} mapped as {list(dataset.class_encodings.values())}", file=f)
        print(f"F1 = {f1:.4f}\nAcc = {acc:.4f}\nMCC = {mcc:.4f}", file=f)

    plot_confusion_matrix(cm, glosses, root_path=pred_path)


def main(params):
    config_path = params.config_path
    log_filename = params.log_filename
    test(config_path, log_filename)


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
