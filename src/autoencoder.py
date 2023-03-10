import os
import sys

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

from src.utils.util import load_gzip
from src.utils.helpers import load_config

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from tqdm import tqdm

from sklearn.metrics import mean_squared_error

from src.utils.pytorch_i3d import InceptionI3d
from src.utils.i3d_data import I3Dataset
from src.utils import spatial_transforms
from src.train_i3d import EarlyStopper

import matplotlib.pyplot as plt
from pathlib import Path
import torchvision


class AutoEncoder(nn.Module):

    def __init__(self, n_layers, k):
        super(AutoEncoder, self).__init__()

        self.k = k
        self.n_layers = n_layers

        if k >= 1024:
            raise ValueError("The bottleneck layer has to have dimension k < 1024")

        # self.encoder = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 4),
        #     nn.ReLU(),
        #     nn.Linear(4, 2),
        # )
        #
        # self.decoder = nn.Sequential(
        #     nn.Linear(2, 4),
        #     nn.ReLU(),
        #     nn.Linear(4, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.Sigmoid()
        # )

        # the autoencoder always has 4 layers, so calculate their input/output dimensions based on the given k
        dim_step = (1024 - self.k) // self.n_layers
        decoder_dims = list(range(self.k, 1024, dim_step))
        decoder_dims.insert(len(decoder_dims), 1024)
        encoder_dims = list(reversed(decoder_dims))

        self.encoder = nn.ModuleList()
        for idx in range(len(encoder_dims) - 1):
            self.encoder.append(nn.Linear(encoder_dims[idx], encoder_dims[idx + 1]))
            if idx + 1 < len(encoder_dims) - 1:  # if not the last layer
                self.encoder.append(nn.ReLU())

        self.decoder = nn.ModuleList()
        for idx in range(len(decoder_dims) - 1):
            self.decoder.append(nn.Linear(decoder_dims[idx], decoder_dims[idx + 1]))
            if idx + 1 < len(decoder_dims) - 1:
                self.decoder.append(nn.ReLU())
            else:  # if last layer
                self.decoder.append(nn.Sigmoid())

    def forward(self, x):
        # x = self.encoder(x)
        # x = self.decoder(x)
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        return x


def plot_training_history(config: dict, training_history: dict, fig_output_root: str):

    train_config = config.get("training")

    specific_glosses = train_config.get("specific_glosses")
    epochs = train_config.get("epochs")
    batch_size = train_config.get("batch_size")
    lr = train_config.get("lr")

    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    filename = f"autoencoder_{glosses_string}_{epochs}_{batch_size}_{lr}"

    # this make sure graphs can be opened in windows
    filename = filename.replace(":", ";")

    epochs = np.arange(1, len(training_history["loss"]["train"]) + 1)

    # clear contents of the plot, to avoid overlap with previous plots
    plt.clf()
    # set the style of the plot
    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    plt.plot(epochs, training_history["loss"]["train"], label="train", ls="--")
    plt.plot(epochs, training_history["loss"]["val"], label="val", ls="-")
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()

    os.makedirs(fig_output_root, exist_ok=True)
    plt.savefig(os.path.join(fig_output_root, filename + '_loss.png'))

    plt.clf()

    for metric in training_history["metric"].keys():
        if "train" in metric:
            plt.plot(epochs, training_history["metric"][metric], label=metric, linestyle='--')
        elif "val" in metric:
            plt.plot(epochs, training_history["metric"][metric], label=metric, linestyle='-')

    plt.legend(loc="best")
    plt.ylabel("Metric")
    plt.xlabel("Epoch")
    plt.tight_layout()

    plt.savefig(os.path.join(fig_output_root, filename + '_metrics.png'))


def train_autoencoder(config: dict, dataloaders: dict, k: int, fig_output_root: str, log_file: str):
    train_config = config.get("training")

    lr = train_config.get("lr")
    weight_decay = train_config.get("weight_decay")
    epochs = train_config.get("epochs")
    n_layers = train_config.get("n_layers")
    use_cuda = train_config.get("use_cuda")

    # initialize autoencoder
    autoencoder = AutoEncoder(n_layers=n_layers, k=k)
    if use_cuda:
        autoencoder.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)

    # early stopping setup
    min_delta = 0.0
    patience = 10
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=min_delta)

    training_history = {"loss": {"train": [], "val": []},
                        "metric": {"train_mse": [], "val_mse": []}}

    min_loss = np.inf
    early_stop_flag = False

    print(f"Training autoencoder with {n_layers} layers and k={k}...")
    for epoch in range(epochs):
        if early_stop_flag:
            print(f"Early stopping: validation loss did not decrease more than {early_stopper.min_delta} in {early_stopper.patience} epochs.")
            break
        for phase in ["train", "val"]:
            tot_loss = 0.0
            num_iter = 0
            epoch_mse = []

            autoencoder.train(True) if phase == "train" else autoencoder.train(False)

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for data in tepoch:
                    num_iter += 1
                    optimizer.zero_grad()

                    feature_vector = Variable(data.cuda())
                    preds = autoencoder(feature_vector)

                    epoch_mse.append(mean_squared_error(y_true=feature_vector.detach().cpu().numpy(), y_pred=preds.detach().cpu().numpy()))

                    loss = criterion(preds, feature_vector)
                    loss.backward()
                    tot_loss += loss.item()

                    if phase == 'train':
                        optimizer.step()
                        tepoch.set_postfix(loss=round(tot_loss / num_iter, 4),
                                           mse=round(np.mean(epoch_mse), 4))

            if phase == "train":
                training_history["loss"]["train"].append(tot_loss/num_iter)
                training_history["metric"]["train_mse"].append(np.mean(epoch_mse))

                # store the model state_dict to store it later if the val loss improves
                train_ckpt = autoencoder.state_dict()

            if phase == "val":
                training_history["loss"]["val"].append(tot_loss / num_iter)
                training_history["metric"]["val_mse"].append(np.mean(epoch_mse))

                early_stop_flag = early_stopper(tot_loss / num_iter)
                if (tot_loss / num_iter) < min_loss:
                    min_loss = tot_loss / num_iter
                    best_ckpt = train_ckpt
                    best_epoch = epoch

                lr_sched.step(tot_loss / num_iter)

    plot_training_history(config, training_history, fig_output_root)

    # make sure we use the best checkpoint
    autoencoder.load_state_dict(best_ckpt)
    autoencoder.train(False)

    print(f"Testing autoencoder with epoch {best_epoch+1} checkpoint...")
    MSE = []
    total_true = []
    total_pred = []
    for feature_vector in dataloaders["train"]:
        feature_vector = Variable(feature_vector.cuda())
        preds = autoencoder(feature_vector)
        total_true.extend(feature_vector.detach().cpu().numpy())
        total_pred.extend(preds.detach().cpu().numpy())

    # print to console
    print(f"Autoencoder k={k}, test MSE={mean_squared_error(y_true=total_true, y_pred=total_pred):.4f}")
    # print to file
    print(f"Autoencoder k={k}, test MSE={mean_squared_error(y_true=total_true, y_pred=total_pred):.4f}", file=log_file)


def train(config: dict, fig_output_root: str, log_output_root: str):

    train_config = config.get("training")
    data_config = config.get("data")

    # parameters for autoencoder training
    specific_glosses = train_config.get("specific_glosses")
    ks = train_config.get("ks")
    lr = train_config.get("lr")
    weight_decay = train_config.get("weight_decay")
    epochs = train_config.get("epochs")
    batch_size = train_config.get("batch_size")

    # parameters to load weights
    run_name = train_config.get("run_name")
    run_epochs = train_config.get("run_epochs")
    run_batch_size = train_config.get("run_batch_size")
    run_lr = train_config.get("run_lr")
    run_optimizer = train_config.get("run_optimizer")
    saved_model_root = train_config.get("saved_model_root")
    use_cuda = train_config.get("use_cuda")
    random_seed = train_config.get("random_seed")

    root = data_config.get("root")
    cngt_clips_folder = data_config.get("cngt_clips_folder")
    signbank_folder = data_config.get("signbank_folder")
    sb_vocab_file = data_config.get("sb_vocab_file")
    window_size = data_config.get("window_size")
    loading_mode = data_config.get("loading_mode")
    input_size = data_config.get("input_size")
    clips_per_class = data_config.get("clips_per_class")

    cngt_root = os.path.join(root, cngt_clips_folder)
    sb_root = os.path.join(root, signbank_folder)
    sb_vocab_path = os.path.join(root, sb_vocab_file)

    # get directory and filename for the checkpoints
    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    run_dir = f"{run_name}_{glosses_string}_{run_epochs}_{run_batch_size}_{run_lr}_{run_optimizer}"

    ckpt_files = [file for file in os.listdir(os.path.join(saved_model_root, run_dir)) if file.endswith(".pt")]
    # load the last saved checkpoint, which contains the minimum val loss
    ckpt_filename = ckpt_files[-1]
    # ckpt_filename = f"i3d_{str(ckpt_epoch).zfill(len(str(run_epochs)))}.pt"

    num_top_glosses = None

    sb_vocab = load_gzip(sb_vocab_path)
    gloss_to_id = sb_vocab['gloss_to_id']

    specific_gloss_ids = [gloss_to_id[gloss] for gloss in specific_glosses]

    cropped_input_size = input_size * 0.875

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomPerspective(),
        torchvision.transforms.RandomAffine(degrees=10),
        torchvision.transforms.RandomHorizontalFlip(),
        spatial_transforms.ColorJitter(num_in_frames=window_size),
        torchvision.transforms.RandomCrop(cropped_input_size)])

    val_transforms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(cropped_input_size)])

    transforms = {"train": train_transforms, "val": val_transforms}
    dataloaders = {}

    for fold in ["train", "val"]:
        print(f"Loading {fold} split...")
        dataset = I3Dataset(loading_mode=loading_mode,
                            cngt_root=cngt_root,
                            sb_root=sb_root,
                            sb_vocab_path=sb_vocab_path,
                            mode="rgb",
                            split=fold,
                            window_size=window_size,
                            transforms=transforms[fold],
                            filter_num=num_top_glosses,
                            specific_gloss_ids=specific_gloss_ids,
                            clips_per_class=clips_per_class,
                            random_seed=random_seed)

        dataloaders[fold] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    i3d = InceptionI3d(num_classes=len(dataset.class_encodings),
                       in_channels=3,
                       window_size=window_size,
                       input_size=cropped_input_size)

    if use_cuda:
        i3d.load_state_dict(torch.load(os.path.join(saved_model_root, run_dir, ckpt_filename)))
    else:
        i3d.load_state_dict(torch.load(os.path.join(saved_model_root, run_dir, ckpt_filename), map_location=torch.device('cpu')))

    if use_cuda:
        i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode



    print(f"Running datapoints through model...")
    with torch.no_grad():  # this deactivates gradient calculations, reducing memory consumption by A LOT
        for phase in ["train", "val"]:
            X_features = torch.zeros((1, 1024))
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for data in tepoch:
                    # get the inputs
                    inputs, labels, _ = data
                    if use_cuda:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())

                    # get the features of the penultimate layer
                    features = i3d.extract_features(inputs)
                    features = torch.squeeze(features, -1)

                    # if X_features is 0 (empty)
                    if X_features.sum() == 0:
                        X_features = features.squeeze()
                    else:
                        # if the last batch has only 1 video, the squeeze function removes an extra dimension and cannot be concatenated
                        if len(features.squeeze().size()) == 1:
                            X_features = torch.cat((X_features, torch.unsqueeze(features.squeeze(), 0)), dim=0)
                        else:
                            X_features = torch.cat((X_features, features.squeeze()), dim=0)

            if phase == "train":
                X_train_features = X_features
            elif phase == "val":
                X_val_features = X_features

    # this normalizes the features in order for them to have values in [0,1]
    train_features = X_train_features / X_train_features.amax(dim=1, keepdim=True)
    val_features = X_val_features / X_val_features.amax(dim=1, keepdim=True)

    dataloaders["train"] = torch.utils.data.DataLoader(train_features, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloaders["val"] = torch.utils.data.DataLoader(val_features, batch_size=batch_size, shuffle=True, num_workers=0)

    with open(os.path.join(log_output_root, "autoencoder_test_log.txt"), "w") as file:
        for k in ks:
            train_autoencoder(config, dataloaders, k, fig_output_root, log_file=file)


def main(params):
    config_path = params.config_path
    fig_output_root = params.fig_output_root
    log_output_root = params.log_output_root

    config = load_config(config_path)

    train(config, fig_output_root, log_output_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
    )

    parser.add_argument(
        "--fig_output_root",
        type=str,
    )

    parser.add_argument(
        "--log_output_root",
        type=str,
    )

    params, _ = parser.parse_known_args()
    main(params)
