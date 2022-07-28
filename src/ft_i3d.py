import os
import sys

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

from src.utils import videotransforms
from src.utils.helpers import load_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-mode', type=str, help='rgb or flow')
# parser.add_argument('-save_model', type=str)
# parser.add_argument('-root', type=str)
#
# args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import src.utils.videotransforms

import numpy as np

from time import sleep
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.utils.pytorch_i3d import InceptionI3d
from src.utils.i3d_data import I3Dataset
from src.utils.resnet import r2plus1d_18
from src.utils.util import extract_zip
from src.utils.loss import f1_loss


# def f1_score(TN, TP, FP, FN):
#     acc = (TP + TN) / (TN + TP + FN + FP)
#     precision = TP / (TP + FP + 1e-6)
#     recall = TP / (TP + FN + 1e-6)
#     F1 = 2 * precision * recall / (precision + recall + 1e-6)
#     return acc, F1, precision, recall


def get_prediction_measures(labels, frame_logits):

    labels = labels.detach().cpu().numpy()
    frame_logits = frame_logits.detach().cpu().numpy()
    FP = 0
    FN = 0
    TP = 0
    TN = 0

    for batch in range(np.shape(labels)[0]):

        y_pred = np.argmax(frame_logits[batch], axis=0)
        y_true = np.argmax(labels[batch], axis=0)

        conf_matrix = confusion_matrix(y_true, y_pred)
        FP += np.sum(conf_matrix.sum(axis=0) - np.diag(conf_matrix))
        FN += np.sum(conf_matrix.sum(axis=1) - np.diag(conf_matrix))
        TP += np.sum(np.diag(conf_matrix))
        TN += conf_matrix.sum() - (FP + FN + TP)

        # for frame in range(np.shape(labels)[2]):
        #     confusion_matrix(labels[batch, :, frame], frame_logits[batch, :, frame])
        #     f_FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        #     f_FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        #     f_TP = np.diag(confusion_matrix)
        #     f_TN = confusion_matrix.sum() - (f_FP + f_FN + f_TP)
        #
        #     FP += f_FP
        #     FN += f_FN
        #     TP += f_TP
        #     TN += f_TN

    # preds = np.argmax(frame_logits.detach().cpu().numpy(), axis=1)
    # gts = np.argmax(labels.detach().cpu().numpy(), axis=1)
    #
    #
    # TP = (preds & gts).sum()
    # TN = ((~preds) & (~gts)).sum()
    # FP = (preds & (~gts)).sum()
    # FN = ((~preds) & gts).sum()

    # batch_f1 = []
    # window_f1 = []
    # for batch in range(labels.size(0)):  # batch
    #     for frame in range(labels.size(2)):
    #         one_hot = torch.zeros(labels[batch, :, frame].shape)
    #         one_hot[torch.topk(frame_logits[batch, :, frame], 1).indices] = 1
    #         window_f1.append(f1_loss(labels[batch, :, frame], one_hot.cuda()))
    #     batch_f1.append(window_f1)

    return TP, TN, FP, FN


def get_class_encodings(cngt_gloss_ids, sb_gloss_ids):
    classes = list(set(cngt_gloss_ids).union(set(sb_gloss_ids)))

    class_to_idx = {}
    for i in range(len(classes)):
        class_to_idx[classes[i]] = i

    return class_to_idx


def run(cfg_path, mode='rgb'):
    cfg = load_config(cfg_path)
    training_cfg = cfg.get("training")

    # training configs
    epochs = training_cfg.get("epochs")
    init_lr = training_cfg.get("init_lr")
    batch_size = training_cfg.get("batch_size")
    save_model = training_cfg.get("model_dir")
    weights_dir = training_cfg.get("weights_dir")

    # data configs
    cngt_zip = cfg.get("data").get("cngt_clips_path")
    sb_zip = cfg.get("data").get("signbank_path")
    window_size = cfg.get("data").get("window_size")

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    val_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    filter_top_glosses = 400  # should be None if no filtering wanted

    print("Loading training split...")
    train_dataset = I3Dataset(cngt_zip, sb_zip, mode, 'train', window_size, train_transforms, filter_top_glosses)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   pin_memory=True)

    print("Loading val split...")
    val_dataset = I3Dataset(cngt_zip, sb_zip, mode, 'val', window_size, val_transforms, filter_top_glosses)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    # datasets = {'train': train_dataset, 'val': val_dataset}

    print("Setting up the model...")
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(weights_dir + '/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load(weights_dir + '/rgb_imagenet.pt'))

    i3d.replace_logits(len(train_dataset.class_encodings))

    # model = r2plus1d_18(pretrained=True, progress=True)

    # prints number of parameters
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    for param in i3d.parameters():
        param.requires_grad = False
    # freeze all layers for fine-tuning

    # unfreeze the ones we want
    i3d.logits.requires_grad_(True)

    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    # optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=0.0000001)
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 1  # accumulate gradient
    steps = 0
    # train it
    for epoch in range(epochs):
        # print('Epoch {}/{}'.format(steps, max_steps))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            min_loss = np.inf
            TP = 0
            TN = 0
            FP = 0
            FN = 0

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs} -- ")
                    num_iter += 1
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    inputs = Variable(inputs.cuda())
                    t = inputs.size(2)
                    labels = Variable(labels.cuda())

                    per_frame_logits = i3d(inputs)
                    # upsample to input size
                    per_frame_logits = F.interpolate(per_frame_logits, size=t, mode='linear')

                    # compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                    tot_loc_loss += loc_loss.item()

                    # compute classification loss (with max-pooling along time B x C x T)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                                  torch.max(labels, dim=2)[0])
                    tot_cls_loss += cls_loss.item()

                    loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                    tot_loss += loss.item()
                    loss.backward()

                    y_pred = np.argmax(per_frame_logits.detach().cpu().numpy(), axis=1)
                    y_true = np.argmax(labels.detach().cpu().numpy(), axis=1)

                    batch_acc = [accuracy_score(y_true[i], y_pred[i]) for i in range(np.shape(y_pred)[0])]
                    batch_f1 = [f1_loss(y_true[i], y_pred[i], average='macro') for i in range(np.shape(y_pred)[0])]

                    # b_TP, b_TN, b_FP, b_FN = get_prediction_measures(labels, per_frame_logits)
                    # batch_acc, batch_f1, _, _ = f1_score(b_TP, b_TN, b_FP, b_FN)
                    # TP += b_TP
                    # TN += b_TN
                    # FP += b_FP
                    # FN += b_FN

                    if num_iter == num_steps_per_update and phase == 'train':
                        steps += 1
                        num_iter = 0
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_sched.step()
                        if steps % 10 == 0:

                            # total_acc, total_f1, _, _ = f1_score(TP, TN, FP, FN)

                            tepoch.set_postfix(loc_loss=round(tot_loc_loss / (10 * num_steps_per_update), 4),
                                               cls_loss=round(tot_cls_loss / (10 * num_steps_per_update), 4),
                                               loss=round(tot_loss / 10, 4),
                                               batch_acc=round(batch_acc, 4),
                                               batch_f1=round(batch_f1, 4))
                                               # total_acc=round(total_acc, 4),
                                               # total_f1=round(total_f1, 4))
                            # print('{} Loc Loss: {:.4f} Cls Loss: {:.4f}\tTot Loss: {:.4f}'.format(phase, tot_loc_loss / (
                            #             10 * num_steps_per_update), tot_cls_loss / (10 * num_steps_per_update),
                            #                                                                       tot_loss / 10))

                            # save the model only if the loss is better
                            if tot_loss < min_loss:
                                min_loss = tot_loss
                                # save model
                                torch.save(i3d.module.state_dict(),
                                           save_model + '/' + 'i3d_' + str(epoch).zfill(len(str(epochs))) + '_' + str(
                                               steps).zfill(6) + '.pt')
                            tot_loss = tot_loc_loss = tot_cls_loss = 0.

                if phase == 'val':
                    val_acc, val_f1, _, _ = f1_score(TP, TN, FP, TN)
                    print(f'Epoch {epoch + 1} validation phase:\n'
                          f'Loc Loss: {tot_loc_loss / num_iter:.4f}\n'
                          f'Cls Loss: {tot_cls_loss / num_iter:.4f}\n'
                          f'Tot Loss: {(tot_loss * num_steps_per_update) / num_iter:.4f}\n'
                          f'Acc: {val_acc:.4f}\n'
                          f'F1: {val_f1:.4f} ')


def main(params):
    config_path = params.config_path
    run(config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
    )
    params, _ = parser.parse_known_args()
    main(params)
