import os
import sys

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

from src.utils import videotransforms
from src.utils.helpers import load_config
from src.utils.util import make_dir

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms
import src.utils.videotransforms

import numpy as np

from time import sleep
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

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

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     # print(cm)
#
#     figure = plt.figure(figsize=(8, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#     return figure


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
    print("Configuring model and parameters...")
    cfg = load_config(cfg_path)
    training_cfg = cfg.get("training")

    # training configs
    epochs = training_cfg.get("epochs")
    init_lr = training_cfg.get("init_lr")
    batch_size = training_cfg.get("batch_size")
    save_model_root = training_cfg.get("model_dir")
    weights_dir = training_cfg.get("weights_dir")

    # data configs
    cngt_zip = cfg.get("data").get("cngt_clips_path")
    sb_zip = cfg.get("data").get("signbank_path")
    window_size = cfg.get("data").get("window_size")

    print(f"Using window size of {window_size} frames")

    # TRANSFORMS BASED ON TORCHVISION IMPLEMENTATION

    train_transforms = transforms.Compose([torchvision.transforms.RandomCrop(224),
                                           torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                           torchvision.transforms.RandomPerspective(p=0.5),
                                           torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                              saturation=0.2, hue=0.2),
                                           torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                           ])

    val_transforms = transforms.Compose([torchvision.transforms.CenterCrop(224)])

    # CUSTOM TRANSFORMS FROM THE GITHUB REPOSITORY
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    val_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    num_top_glosses = 2  # should be None if no filtering wanted

    print("Loading training split...")
    train_dataset = I3Dataset(cngt_zip, sb_zip, mode, 'train', window_size, transforms=train_transforms, filter_num=num_top_glosses)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print("Loading val split...")
    val_dataset = I3Dataset(cngt_zip, sb_zip, mode, 'val', window_size, transforms=val_transforms, filter_num=num_top_glosses)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    # datasets = {'train': train_dataset, 'val': val_dataset}

    print("Setting up the model...")
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(weights_dir + '/flow_imagenet.pt'))
    else:
        # i3d = InceptionI3d(400, in_channels=3)
        # i3d.load_state_dict(torch.load(weights_dir + '/rgb_imagenet.pt'))
        i3d = InceptionI3d(157, in_channels=3, window_size=16, input_size=224)
        i3d.load_state_dict(torch.load(weights_dir + '/rgb_charades.pt'))

        # THIS LINE IS ADDED TO TRAIN FROM SCRATCH
        # i3d = InceptionI3d(2, in_channels=3, window_size=window_size, input_size=224)

    i3d.replace_logits(len(train_dataset.class_encodings))

    print(f"The model has {len(train_dataset.class_encodings)} classes")

    # model = r2plus1d_18(pretrained=True, progress=True)

    # prints number of parameters
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    n_layers = 0
    # freeze all layers for fine-tuning
    for param in i3d.parameters():
        param.requires_grad = False
        n_layers += 1

    # unfreeze the ones we want
    i3d.logits.requires_grad_(True)
    # layers are ['Mixed_5c', 'Mixed_5b', 'MaxPool3d_5a_2x2', 'Mixed_4f', 'Mixed_4e', 'Mixed_4d', 'Mixed_4c', 'Mixed_4b']
    unfreeze_layers = []
    for layer in unfreeze_layers:
        i3d.end_points[layer].requires_grad_(True)

    print(f"The last {len(unfreeze_layers) + 1} out of 17 blocks are unfrozen.")

    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    # optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=0.0000001)
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    # lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # just before the actual training loop, create a file where the training log will be saved
    writer = SummaryWriter()

    # before starting the train loop, make sure the directory where the model will be stored is created/exists
    new_save_dir = f'b{str(batch_size)}_{str(optimizer).split("(")[0].strip()}_lr{str(lr)}_ep{str(epochs)}'
    save_model_dir = os.path.join(save_model_root, new_save_dir)
    make_dir(save_model_dir)

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
            min_loss = np.inf
            print_freq = 1

            acc_list = []
            f1_list = []

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs} -- ")
                    num_iter += 1
                    # get the inputs
                    inputs, labels = data

                    optimizer.zero_grad()  # clear gradients

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
                    cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        torch.max(per_frame_logits, dim=2)[0],
                        torch.max(labels, dim=2)[0])
                    tot_cls_loss += cls_loss.item()

                    loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                    tot_loss += loss.item()
                    loss.backward()

                    tot_loss += loss.item()

                    # get batch accuracy and f1
                    y_pred = np.argmax(per_frame_logits.detach().cpu().numpy(), axis=1)
                    y_true = np.argmax(labels.detach().cpu().numpy(), axis=1)

                    batch_acc = np.mean([accuracy_score(y_true[i], y_pred[i]) for i in range(np.shape(y_pred)[0])])
                    batch_f1 = np.mean(
                        [f1_score(y_true[i], y_pred[i], average='macro') for i in range(np.shape(y_pred)[0])])

                    acc_list.append(batch_acc)
                    f1_list.append(batch_f1)

                    if phase == 'train' and num_iter == num_steps_per_update:

                        writer.add_scalar("train/loss", tot_loss / num_steps_per_update, steps)
                        writer.add_scalar("train/loc_loss", tot_loc_loss / num_steps_per_update, steps)
                        writer.add_scalar("train/cls_loss", tot_cls_loss / num_steps_per_update, steps)
                        writer.add_scalar("train/acc", np.mean(acc_list), steps)
                        writer.add_scalar("train/f1", np.mean(f1_list), steps)

                        optimizer.step()
                        steps += 1
                        num_iter = 0

                        if steps % print_freq == 0:
                            tepoch.set_postfix(loss=round(tot_loss / print_freq, 4),
                                               loc_loss=round(tot_loc_loss / print_freq, 4),
                                               cls_loss=round(tot_cls_loss / print_freq, 4),
                                               total_acc=round(np.mean(acc_list), 4),
                                               total_f1=round(np.mean(f1_list), 4))

                        # save model only when loss is lower than the minimum loss
                        if tot_loss < min_loss:
                            min_loss = tot_loss
                            # save model
                            torch.save(i3d.module.state_dict(),
                                       save_model_dir + '/' + 'i3d_' + str(epoch).zfill(len(str(epochs))) + '_' + str(
                                        num_iter).zfill(6) + '.pt')

                        tot_loss = 0.0
                        loc_loss = 0.0
                        cls_loss = 0.0

                # after processing the data
                if phase == 'val':
                    lr_sched.step(tot_loss)

                    writer.add_scalar("val/loss", tot_loss / num_iter, steps)
                    writer.add_scalar("val/loc_loss", tot_loc_loss / num_iter, steps)
                    writer.add_scalar("val/cls_loss", tot_cls_loss / num_iter, steps)
                    writer.add_scalar("val/acc", np.mean(acc_list), steps)
                    writer.add_scalar("val/f1", np.mean(f1_list), steps)

                    print('-------------------------\n'
                          f'Epoch {epoch + 1} validation phase:\n'
                          f'Tot Loss: {tot_loss / num_iter:.4f}\t'
                          f'Loc Loss: {tot_loc_loss / num_iter:.4f}\t'
                          f'Cls Loss: {tot_cls_loss / num_iter:.4f}\t'
                          f'Acc: {np.mean(acc_list):.4f}\t'
                          f'F1: {np.mean(f1_list):.4f}\n'
                          '-------------------------\n')

        writer.close()


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
