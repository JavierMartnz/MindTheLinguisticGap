import os
import sys
import json

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

from src.utils import videotransforms
from src.utils.helpers import load_config
from src.utils.util import load_gzip

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
from torchvision import datasets, transforms
from torchsummary import summary

import numpy as np

from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.utils.pytorch_i3d import InceptionI3d
from src.utils.i3d_dimensions_exp import InceptionI3d as InceptionDims
from src.utils.i3d_dimensions_conv import InceptionI3d as InceptionDimsConv
from src.utils.i3d_data import I3Dataset
from src.utils import spatial_transforms


def run(cfg_path, mode='rgb'):
    print("Configuring model and parameters...")
    cfg = load_config(cfg_path)
    training_cfg = cfg.get("training")
    data_cfg = cfg.get("data")

    # training configs
    extra_conv = training_cfg.get("extra_conv")
    model_weights = training_cfg.get("model_weights")
    specific_glosses = training_cfg.get("specific_glosses")
    run_name = training_cfg.get("run_name")
    epochs = training_cfg.get("epochs")
    init_lr = training_cfg.get("init_lr")
    batch_size = training_cfg.get("batch_size")
    save_model_root = training_cfg.get("model_dir")
    weights_dir = training_cfg.get("weights_dir")

    # data configs
    cngt_zip = data_cfg.get("cngt_clips_path")
    sb_zip = data_cfg.get("signbank_path")
    sb_vocab_path = data_cfg.get("sb_vocab_path")
    window_size = data_cfg.get("window_size")
    loading_mode = data_cfg.get("data_loading")
    use_diag_videos = data_cfg.get("use_diag_videos")
    diagonal_videos_path = data_cfg.get("diagonal_videos_path") if use_diag_videos else None
    final_pooling_size = data_cfg.get("final_pooling_size")
    input_size = data_cfg.get("input_size")

    if extra_conv:
        assert model_weights

    print(f"Using window size of {window_size} frames")
    print(f"Input size is {input_size}")

    cropped_input_size = input_size * 0.875

    train_transforms = transforms.Compose([
        transforms.RandomPerspective(),
        transforms.RandomAffine(degrees=10),
        transforms.RandomHorizontalFlip(),
        spatial_transforms.ColorJitter(num_in_frames=window_size),
        transforms.RandomCrop(cropped_input_size)])

    # validation transforms should never contain any randomness
    val_transforms = transforms.Compose([transforms.CenterCrop(cropped_input_size)])

    num_top_glosses = None  # should be None if no filtering wanted

    # get glosses from the class encodings
    sb_vocab = load_gzip(sb_vocab_path)
    gloss_to_id = sb_vocab['gloss_to_id']

    specific_gloss_ids = [gloss_to_id[gloss] for gloss in specific_glosses]

    print("Loading training split...")
    train_dataset = I3Dataset(loading_mode, cngt_zip, sb_zip, sb_vocab_path, mode, 'train', window_size,
                              transforms=train_transforms, filter_num=num_top_glosses, specific_gloss_ids=specific_gloss_ids,
                              diagonal_videos_path=diagonal_videos_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print("Loading val split...")
    val_dataset = I3Dataset(loading_mode, cngt_zip, sb_zip, sb_vocab_path, mode, 'val', window_size,
                            transforms=val_transforms, filter_num=num_top_glosses, specific_gloss_ids=specific_gloss_ids)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    print("Setting up the model...")
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(weights_dir + '/flow_imagenet.pt'))
    else:
        # THIS IS THE STANDARD ORIGINAL I3D
        i3d = InceptionI3d(400, in_channels=3, window_size=window_size, input_size=cropped_input_size)

        # i3d.load_state_dict(torch.load(weights_dir + '/rgb_charades.pt'))
        i3d.load_state_dict(torch.load(weights_dir + '/rgb_imagenet.pt'))

    # changes the last layer in order to accommodate the new number of classes (after loading weights)
    i3d.replace_logits(num_classes=len(train_dataset.class_encodings))

    # this only applies to the network with the extra conv layer
    if extra_conv:
        i3d.add_dim_conv()

    print(f"\tThe model has {len(train_dataset.class_encodings)} classes")

    n_layers = 0
    # freeze all layers for fine-tuning
    for param in i3d.parameters():
        param.requires_grad = False
        n_layers += 1

    # unfreeze the layers that we want to train
    if extra_conv:
        i3d.dims_conv.requires_grad_(True)

    i3d.logits.requires_grad_(True)

    # layers are ['Mixed_5c', 'Mixed_5b', 'MaxPool3d_5a_2x2', 'Mixed_4f', 'Mixed_4e', 'Mixed_4d', 'Mixed_4c', 'Mixed_4b']
    unfreeze_layers = []
    for layer in unfreeze_layers:
        i3d.end_points[layer].requires_grad_(True)

    if extra_conv:
        print(f"\tThe last {len(unfreeze_layers) + 2} out of 18 blocks are unfrozen.")
    else:
        print(f"\tThe last {len(unfreeze_layers) + 1} out of 17 blocks are unfrozen.")

    # prints number of parameters
    trainable_params = sum(p.numel() for p in i3d.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in i3d.parameters())
    print(f"\tThe network has {trainable_params} trainable parameters out of {total_params}")

    i3d.cuda()

    # print summary of the network, similar to keras
    # summary(i3d, (3, 16, 224, 224))

    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)

    lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # before starting the training loop, make sure the directory where the model will be stored is created/exists
    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    new_save_dir = f"{run_name}_{glosses_string}_{epochs}_{batch_size}_{lr}_{str(optimizer).split('(')[0].strip()}"
    save_model_dir = os.path.join(save_model_root, new_save_dir)
    os.makedirs(save_model_dir, exist_ok=True)

    steps = 0  # count the number of optimizer steps

    # THESE PARAMETERS CAN BE CHANGED
    num_steps_per_update = 1  # number of batches for which the gradient accumulates before backpropagation
    print_freq = 1  # number of optimizer steps before printing batch loss and metrics

    training_history = {'train_loss': [], 'train_accuracy': [], 'train_f1': [],
                        'val_loss': [], 'val_accuracy': [], 'val_f1': []}

    # start training
    for epoch in range(epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            num_iter = 0  # count number of iterations in an epoch
            num_acc_loss = 0  # count number of iterations in which the model hasn't been stored
            min_loss = np.inf

            acc_list = []
            f1_list = []

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs} -- ")
                    num_iter += 1
                    num_acc_loss += 1

                    # clear gradients
                    optimizer.zero_grad()

                    # get the inputs
                    inputs, labels, _ = data
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                    # forward pass of the inputs through the network
                    sign_logits = i3d(inputs)
                    sign_logits = torch.squeeze(sign_logits, -1)

                    # calculate and backpropagate the loss
                    tot_loss = F.binary_cross_entropy_with_logits(sign_logits, labels)
                    tot_loss.backward()

                    # save the loss
                    tot_loss = tot_loss.item()

                    # get predictions and append them for later
                    y_pred = np.argmax(sign_logits.detach().cpu().numpy(), axis=1)
                    y_true = np.argmax(labels.detach().cpu().numpy(), axis=1)

                    acc_list.append(accuracy_score(y_true.flatten(), y_pred.flatten()))
                    f1_list.append(f1_score(y_true.flatten(), y_pred.flatten()))

                    # this if clause allows gradient accumulation and saves the model weights if the loss improves
                    if phase == 'train' and num_iter % num_steps_per_update == 0:
                        optimizer.step()
                        steps += 1
                        num_acc_loss += 1

                        if steps % print_freq == 0:
                            tepoch.set_postfix(loss=round(tot_loss / num_acc_loss, 4),
                                               total_acc=round(np.mean(acc_list), 4),
                                               total_f1=round(np.mean(f1_list), 4))

                # after processing all data for this epoch, we store the loss and metrics and the model weight if the
                # loss decreased wrt last epoch
                if phase == 'train':
                    # add values to training history
                    training_history['train_loss'].append(tot_loss / num_acc_loss)
                    training_history['train_accuracy'].append(np.mean(acc_list))
                    training_history['train_f1'].append(np.mean(f1_list))

                    # save model only when total loss is lower than the minimum loss achieved so far
                    if tot_loss < min_loss:
                        min_loss = tot_loss
                        # save model
                        torch.save(i3d.module.state_dict(),
                                   save_model_dir + '/' + 'i3d_' + str(epoch).zfill(len(str(epochs))) + '.pt')
                        # reset losses and counter
                        num_acc_loss = 0
                        tot_loss = 0.0

                # after processing the data, record validation metrics
                elif phase == 'val':
                    training_history['val_loss'].append(tot_loss / num_iter)
                    training_history['val_accuracy'].append(np.mean(acc_list))
                    training_history['val_f1'].append(np.mean(f1_list))

                    print('-------------------------\n'
                          f'Epoch {epoch + 1} validation phase:\n'
                          f'Loss: {tot_loss / num_iter:.4f}\t'
                          f'Acc: {np.mean(acc_list):.4f}\t'
                          f'F1: {np.mean(f1_list):.4f}\n'
                          '-------------------------\n')

        # make the scheduler step only at the end of the epoch
        lr_sched.step()

    with open(os.path.join(save_model_dir, 'training_history.txt'), 'w') as file:
        file.write(json.dumps(training_history))


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
