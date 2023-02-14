import os
import sys

import cv2

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import matthews_corrcoef as MCC
import matplotlib.pyplot as plt
import itertools
from torchvision.utils import save_image
from torchvision.io import write_video

from src.utils.i3d_data import I3Dataset
from src.utils.helpers import load_config, make_dir
from src.utils.pytorch_i3d import InceptionI3d
from src.utils.i3d_dimensions_exp import InceptionI3d as InceptionDims
from src.utils.i3d_dimensions_conv import InceptionI3d as InceptionDimsConv
from src.utils.util import load_gzip, save_gzip
from torchsummary import summary
from torchvision import transforms


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
    batch_size = test_cfg.get("batch_size")
    use_cuda = test_cfg.get("use_cuda")
    specific_glosses = test_cfg.get("specific_glosses")

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
    save_predictions = data_cfg.get("save_predictions")
    input_size = data_cfg.get("input_size")

    # get directory and filename for the checkpoints
    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    run_dir = f"{run_name}_{glosses_string}_{num_epochs}_{run_batch_size}_{learning_rate}_{optimizer}"
    # run_dir = f"b{run_batch_size}_{optimizer}_lr{learning_rate}_ep{num_epochs}_{run_name}"
    ckpt_filename = f"i3d_{str(ckpt_epoch).zfill(len(str(num_epochs)))}.pt"
    ckpt_folder = ckpt_filename.split('.')[0]

    if use_diag_videos:
        fold += "_diag"

    pred_path = os.path.join(pred_dir, run_dir, ckpt_folder, fold)
    make_dir(pred_path)
    if save_predictions:
        make_dir(os.path.join(pred_path, "TP"))
        make_dir(os.path.join(pred_path, "TN"))
        make_dir(os.path.join(pred_path, "FP"))
        make_dir(os.path.join(pred_path, "FN"))

    num_top_glosses = None

    # get glosses from the class encodings
    sb_vocab = load_gzip(sb_vocab_path)
    gloss_to_id = sb_vocab['gloss_to_id']
    id_to_gloss = sb_vocab['id_to_gloss']

    specific_gloss_ids = [gloss_to_id[gloss] for gloss in specific_glosses]

    cropped_input_size = input_size * 0.875

    test_transforms = transforms.Compose([transforms.CenterCrop(cropped_input_size)])

    print(f"Loading {fold} split...")
    dataset = I3Dataset(loading_mode=loading_mode,
                        cngt_zip=cngt_zip,
                        sb_zip=sb_zip,
                        sb_vocab_path=sb_vocab_path,
                        mode=mode,
                        split=fold,
                        window_size=window_size,
                        transforms=test_transforms,
                        filter_num=num_top_glosses,
                        specific_gloss_ids=specific_gloss_ids,
                        diagonal_videos_path=diag_videos_path)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    glosses = []

    for gloss_id in dataset.class_encodings.keys():
        glosses.append(id_to_gloss[gloss_id])

    print(f"Predicting for glosses {glosses} mapped as classes {list(dataset.class_encodings.values())}")

    i3d = InceptionI3d(num_classes=len(dataset.class_encodings),
                       in_channels=3,
                       window_size=window_size,
                       input_size=cropped_input_size)

    if use_cuda:
        i3d.load_state_dict(torch.load(os.path.join(model_dir, run_dir, ckpt_filename)))
        i3d.cuda()
    else:
        i3d.load_state_dict(torch.load(os.path.join(model_dir, run_dir, ckpt_filename), map_location=torch.device('cpu')))

    print(f"Successfully loaded model weights from {os.path.join(model_dir, run_dir, ckpt_filename)}")

    i3d.train(False)  # Set model to evaluate mode

    # summary(i3d, (3, 16, 224, 224))

    total_pred = []
    total_true = []
    discard_videos = []
    diagonal_videos = []
    img_cnt = 0

    print(f"Predicting on {fold} set...")
    with torch.no_grad():  # this deactivates gradient calculations, reducing memory consumption by A LOT
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                # get the inputs
                inputs, labels, video_paths = data
                if use_cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                # forward pass of the inputs through the network
                sign_logits = i3d(inputs)
                sign_logits = torch.squeeze(sign_logits, -1)

                # use these for whole sign prediction
                y_pred = np.argmax(sign_logits.detach().cpu().numpy(), axis=1)
                y_true = np.argmax(labels.detach().cpu().numpy(), axis=1)

                # upsample output to input size
                # per_frame_logits = F.interpolate(per_frame_logits, size=inputs.size(2), mode='linear')

                # use these for "frame" prediction
                # y_pred = np.argmax(per_frame_logits.detach().cpu().numpy(), axis=1)
                # y_true = np.argmax(labels.detach().cpu().numpy(), axis=1)

                # save predicitions for later
                if len(total_pred) == 0:
                    total_pred = y_pred.flatten()
                    total_true = y_true.flatten()
                else:
                    total_pred = np.append(total_pred, y_pred.flatten())
                    total_true = np.append(total_true, y_true.flatten())

                # swap to [B, T, H, W, C] for storing
                images = inputs.permute([0, 2, 1, 3, 4])

                for batch in range(images.size(0)):
                    if y_pred.size > 1:
                        pred = y_pred[batch]
                        label = y_true[batch]
                    else:
                        pred = y_pred
                        label = y_true

                    filename = os.path.basename(video_paths[batch])
                    if label == pred and label == 0:  # TP
                        video_path = os.path.join(pred_path, "TP", filename)
                        diagonal_videos.append(filename)
                    elif label == pred and label == 1:  # TN
                        video_path = os.path.join(pred_path, "TN", filename)
                        diagonal_videos.append(filename)
                    elif label != pred and label == 0:  # FN
                        video_path = os.path.join(pred_path, "FN", filename)
                        discard_videos.append(filename)
                    elif label != pred and label == 1:  # FP
                        video_path = os.path.join(pred_path, "FP", filename)
                        discard_videos.append(filename)

                    # change video from [T, C, H, W] to [T, H, W, C] and denormalize
                    if save_predictions:
                        video = images[batch].permute([0, 2, 3, 1]).detach().cpu() * 255.
                        write_video(video_path, video, fps=25)
                    img_cnt += 1

    # save_gzip(discard_videos, os.path.join(pred_path, "discard_list.gzip"))
    # save_gzip(diagonal_videos, os.path.join(pred_path, f"{run_name}_{fold}_diagonal_videos.gzip"))

    acc = accuracy_score(total_true, total_pred)
    p = precision_score(total_true, total_pred)
    r = recall_score(total_true, total_pred)
    f1 = f1_score(total_true, total_pred)
    mcc = MCC(total_true, total_pred)
    cm = confusion_matrix(total_true, total_pred)

    metrics_string = f"Acc = {acc:.4f}\nP = {p:.4f}\nR = {r:.4f}\nF1 = {f1:.4f}\nMCC = {mcc:.4f}"

    print(metrics_string)
    print(cm)

    logfile_path = os.path.join(pred_path, log_filename)

    if os.path.exists(logfile_path):
        os.remove(logfile_path)
    with open(logfile_path, 'w') as f:
        print(f"Predicting for glosses {glosses} mapped as {list(dataset.class_encodings.values())}", file=f)
        print(metrics_string, file=f)

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
