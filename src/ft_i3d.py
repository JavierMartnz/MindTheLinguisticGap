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

from src.utils.pytorch_i3d import InceptionI3d
from src.utils.i3d_data import I3Dataset
from src.utils.util import extract_zip


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
    train_transforms = transforms.Compose([videotransforms.RandomCrop(256),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    val_transforms = transforms.Compose([videotransforms.CenterCrop(256)])

    print("Loading training split...")
    train_dataset = I3Dataset(cngt_zip, sb_zip, mode, 'train', window_size, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                   pin_memory=True)

    print("Loading val split...")
    val_dataset = I3Dataset(cngt_zip, sb_zip, mode, 'val', window_size, val_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    datasets = {'train': train_dataset, 'val': val_dataset}

    print("Setting up the model...")
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(weights_dir + '/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load(weights_dir + '/rgb_imagenet.pt'))

    i3d.replace_logits(len(train_dataset.class_encodings))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 4  # accumulate gradient
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

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"Epoch {str(epoch).zfill(len(str(epochs)))}/{epochs} -- ")
                    num_iter += 1
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    inputs = Variable(inputs.cuda())
                    t = inputs.size(2)
                    labels = Variable(labels.cuda())

                    per_frame_logits = i3d(inputs)
                    # upsample to input size
                    per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                    print(labels.size(), per_frame_logits.size())

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

                    if num_iter == num_steps_per_update and phase == 'train':
                        steps += 1
                        num_iter = 0
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_sched.step()
                        if steps % 10 == 0:
                            tepoch.set_postfix(loc_loss=tot_loc_loss / (10 * num_steps_per_update),
                                               cls_loss=tot_cls_loss / (10 * num_steps_per_update),
                                               loss=tot_loss / 10)
                            # print('{} Loc Loss: {:.4f} Cls Loss: {:.4f}\tTot Loss: {:.4f}'.format(phase, tot_loc_loss / (
                            #             10 * num_steps_per_update), tot_cls_loss / (10 * num_steps_per_update),
                            #                                                                       tot_loss / 10))

                            # save model
                            torch.save(i3d.module.state_dict(),
                                       save_model + '/' + 'i3d_' + str(epoch).zfill(len(str(epochs))) + '_' + str(steps).zfill(6) + '.pt')
                            tot_loss = tot_loc_loss = tot_cls_loss = 0.
                if phase == 'val':
                    print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss / num_iter,
                                                                                         tot_cls_loss / num_iter, (
                                                                                                 tot_loss * num_steps_per_update) / num_iter))


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
