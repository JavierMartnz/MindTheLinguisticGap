import os
import sys
import json

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

from src.utils.helpers import load_config
from src.utils.util import load_gzip

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchsummary import summary

import numpy as np

from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.utils.pytorch_i3d import InceptionI3d
from src.utils.i3d_data import I3Dataset
from src.utils import spatial_transforms
from src.plot_training_history import plot_train_history

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = np.inf

    def __call__(self, val_loss):
        # if loss doesn't improve or improves but less than min_delta
        if val_loss > self.min_val_loss or (val_loss < self.min_val_loss and (self.min_val_loss - val_loss) < self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0
            self.min_val_loss = val_loss
            return False

def train(specific_glosses: list, config: dict, mode='rgb'):
    print(f"\nTraining for {specific_glosses[0]} and {specific_glosses[1]}")
    print("Configuring model and parameters...")

    training_cfg = config.get("training")
    data_cfg = config.get("data")

    # data configs
    root = data_cfg.get("root")
    cngt_clips_folder = data_cfg.get("cngt_clips_folder")
    signbank_folder = data_cfg.get("signbank_folder")
    sb_vocab_file = data_cfg.get("sb_vocab_file")
    window_size = data_cfg.get("window_size")
    loading_mode = data_cfg.get("loading_mode")
    input_size = data_cfg.get("input_size")
    clips_per_class = data_cfg.get("clips_per_class")

    # training configs
    run_name = training_cfg.get("run_name")
    epochs = training_cfg.get("epochs")
    batch_size = training_cfg.get("batch_size")
    init_lr = training_cfg.get("init_lr")
    momentum = training_cfg.get("momentum")
    weight_decay = training_cfg.get("weight_decay")
    save_model_root = training_cfg.get("save_model_root")
    weights_dir_path = training_cfg.get("weights_dir_path")
    use_cuda = training_cfg.get("use_cuda")
    random_seed = training_cfg.get("random_seed")

    # stitch together the paths
    cngt_clips_root = os.path.join(root, cngt_clips_folder)
    sb_root = os.path.join(root, signbank_folder)
    sb_vocab_path = os.path.join(root, sb_vocab_file)

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

    transforms_list = [train_transforms, val_transforms]
    dataloaders = {}

    for i, split in enumerate(["train", "val"]):
        print(f"Loading {split} split...")
        dataset = I3Dataset(loading_mode,
                            cngt_clips_root,
                            sb_root,
                            sb_vocab_path,
                            mode,
                            split,
                            window_size,
                            transforms=transforms_list[i],
                            filter_num=num_top_glosses,
                            specific_gloss_ids=specific_gloss_ids,
                            clips_per_class=clips_per_class,
                            random_seed=random_seed)

        dataloaders[split] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print("Setting up the model...")
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(weights_dir_path + '/flow_imagenet.pt'))
    else:
        # THIS IS THE STANDARD ORIGINAL I3D
        i3d = InceptionI3d(400, in_channels=3, window_size=window_size, input_size=cropped_input_size)

        # i3d.load_state_dict(torch.load(weights_dir + '/rgb_charades.pt'))
        i3d.load_state_dict(torch.load(weights_dir_path + '/rgb_imagenet.pt'))

    # changes the last layer in order to accommodate the new number of classes (after loading weights)
    i3d.replace_logits(num_classes=len(dataset.class_encodings))

    print(f"\tThe model has {len(dataset.class_encodings)} classes")

    n_layers = 0
    # freeze all layers for fine-tuning
    for param in i3d.parameters():
        param.requires_grad = False
        n_layers += 1

    # unfreeze the prev-to-last one
    i3d.logits.requires_grad_(True)

    # layers are ['Mixed_5c', 'Mixed_5b', 'MaxPool3d_5a_2x2', 'Mixed_4f', 'Mixed_4e', 'Mixed_4d', 'Mixed_4c', 'Mixed_4b']
    unfreeze_layers = []
    for layer in unfreeze_layers:
        i3d.end_points[layer].requires_grad_(True)

    print(f"\tThe last {len(unfreeze_layers) + 1} out of 17 blocks are unfrozen.")

    # prints number of parameters
    trainable_params = sum(p.numel() for p in i3d.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in i3d.parameters())
    print(f"\tThe network has {trainable_params} trainable parameters out of {total_params}")

    if use_cuda:
        i3d.cuda()

    # print summary of the network, similar to keras
    # summary(i3d, (3, 16, 224, 224))

    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # early stopping setup
    min_delta = 0.0
    patience = 10
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    # lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=min_delta)

    # before starting the training loop, make sure the directory where the model will be stored is created/exists
    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    new_save_dir = f"{run_name}_{glosses_string}_{epochs}_{batch_size}_{lr}_{str(optimizer).split('(')[0].strip()}"
    save_model_dir = os.path.join(save_model_root, new_save_dir)
    os.makedirs(save_model_dir, exist_ok=True)

    training_history = {'train_loss': [], 'train_accuracy': [], 'train_f1': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}

    min_loss = np.inf
    early_stop_flag = False

    # start training
    for epoch in range(epochs):
        # if the flag was raised in the previous epoch, finish training
        if early_stop_flag:
            print(f"Early stop: validation loss did not decrease more than {early_stopper.min_delta} in {early_stopper.patience} epochs.")
            break

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            num_iter = 0  # count number of iterations in an epoch

            acc_list = []
            f1_list = []

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs} -- ")
                    num_iter += 1

                    # clear gradients
                    optimizer.zero_grad()

                    # get the inputs
                    inputs, labels, _ = data
                    if use_cuda:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())

                    # forward pass of the inputs through the network
                    sign_logits = i3d(inputs)
                    sign_logits = torch.squeeze(sign_logits, -1)

                    # calculate and backpropagate the loss
                    loss = F.binary_cross_entropy_with_logits(sign_logits, labels)
                    loss.backward()

                    # save the loss
                    tot_loss += loss.item()

                    # get predictions and append them for later
                    y_pred = np.argmax(sign_logits.detach().cpu().numpy(), axis=1)
                    y_true = np.argmax(labels.detach().cpu().numpy(), axis=1)

                    acc_list.append(accuracy_score(y_true.flatten(), y_pred.flatten()))
                    f1_list.append(f1_score(y_true.flatten(), y_pred.flatten()))

                    if phase == 'train':
                        optimizer.step()
                        tepoch.set_postfix(loss=round(tot_loss / num_iter, 4),
                                           total_acc=round(np.mean(acc_list), 4),
                                           total_f1=round(np.mean(f1_list), 4))

                # after processing all data for this epoch, we store the loss and metrics and the model weight only if the loss decreased wrt last epoch
                if phase == 'train':
                    # add values to training history
                    training_history['train_loss'].append(tot_loss / num_iter)
                    training_history['train_accuracy'].append(np.mean(acc_list))
                    training_history['train_f1'].append(np.mean(f1_list))

                    # store the model state_dict to store it later if the val loss improves
                    train_ckpt = i3d.module.state_dict()

                # after processing the data, record validation metrics and check for early stopping
                elif phase == 'val':
                    training_history['val_loss'].append(tot_loss / num_iter)
                    training_history['val_accuracy'].append(np.mean(acc_list))
                    training_history['val_f1'].append(np.mean(f1_list))

                    print('-------------------------\n'
                          f'Epoch {epoch + 1} validation phase:\n'
                          f'Loss: {tot_loss / num_iter:.4f}\t'
                          f'Acc: {np.mean(acc_list):.4f}\t'
                          f'F1: {np.mean(f1_list):.4f}\n'
                          '-------------------------')

                    early_stop_flag = early_stopper(tot_loss / num_iter)

                    # save model only when total loss is lower than the minimum loss achieved so far
                    if (tot_loss / num_iter) < min_loss:
                        print(f"Saving checkpoint as val loss was reduced from {round(min_loss, 4)} to {round(tot_loss / num_iter, 4)}\n")
                        min_loss = tot_loss / num_iter
                        # save model
                        torch.save(train_ckpt, save_model_dir + '/' + 'i3d_' + str(epoch).zfill(len(str(epochs))) + '.pt')

                    lr_sched.step(tot_loss / num_iter)

    train_hist_output_root = "/vol/tensusers5/jmartinez/graphs/train_hist"

    plot_train_history(specific_glosses, config, training_history, fig_output_root=train_hist_output_root)

    with open(os.path.join(save_model_dir, 'training_history.txt'), 'w') as file:
        file.write(json.dumps(training_history))


def main(params):
    config_path = params.config_path

    config = load_config(config_path)

    train_config = config.get("training")

    reference_sign = train_config.get("reference_sign")
    train_signs = train_config.get("train_signs")

    assert type(train_signs) == list, "The variable 'train_signs' must be a list."

    for sign in train_signs:
        train(specific_glosses=[reference_sign, sign],
              config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
    )
    params, _ = parser.parse_known_args()
    main(params)
