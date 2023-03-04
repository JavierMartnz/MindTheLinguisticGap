import os
import sys

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

from src.utils import videotransforms
from src.utils.helpers import load_config
from src.utils.util import load_gzip

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
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

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, mean_squared_error

from src.utils.pytorch_i3d import InceptionI3d
from src.utils.i3d_dimensions_exp import InceptionI3d as InceptionDims
from src.utils.i3d_dimensions_conv import InceptionI3d as InceptionDimsConv
from src.utils.i3d_data import I3Dataset
from src.utils import spatial_transforms

import matplotlib.pyplot as plt
from pathlib import Path

class AutoEncoder(nn.Module):

    def __init__(self, k):
        super(AutoEncoder, self).__init__()

        self.k = k
        self.n_layers = 4

        if k >= 1024:
            raise ValueError("The bottlenack layer has to have dimension k < 1024")

        # self.encoder = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     # nn.ReLU(),
        #     # nn.Linear(512, 256),
        #     # nn.ReLU(),
        #     # nn.Linear(256, 128),
        #     # nn.ReLU(),
        #     # nn.Linear(128, 64),
        #     # nn.ReLU(),
        #     # nn.Linear(64, 32),
        #     # nn.ReLU(),
        #     # nn.Linear(32, 16),
        #     # nn.ReLU(),
        #     # nn.Linear(16, 8),
        #     # nn.ReLU(),
        #     # nn.Linear(8, 4),
        #     # nn.ReLU(),
        #     # nn.Linear(4, 2),
        # )
        #
        # self.decoder = nn.Sequential(
        #     # nn.Linear(2, 4),
        #     # nn.ReLU(),
        #     # nn.Linear(4, 8),
        #     # nn.ReLU(),
        #     # nn.Linear(8, 16),
        #     # nn.ReLU(),
        #     # nn.Linear(16, 32),
        #     # nn.ReLU(),
        #     # nn.Linear(32, 64),
        #     # nn.ReLU(),
        #     # nn.Linear(64, 128),
        #     # nn.ReLU(),
        #     # nn.Linear(128, 256),
        #     # nn.ReLU(),
        #     # nn.Linear(256, 512),
        #     # nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.Sigmoid()
        # )

        # the autoencoder always has 4 layers, so calculate their input/output dimensions based on the given k
        dim_step = (1024 - self.k) // self.n_layers
        decoder_dims = list(range(self.k, 1024, dim_step))
        decoder_dims.insert(len(decoder_dims), 1024)
        encoder_dims = list(reversed(decoder_dims))

        self.encoder = nn.ModuleList()
        for idx in range(len(encoder_dims)-1):
            self.encoder.append(nn.Linear(encoder_dims[idx], encoder_dims[idx+1]))
            if idx+1 < len(encoder_dims)-1:  # if not the last layer
                self.encoder.append(nn.ReLU())

        self.decoder = nn.ModuleList()
        for idx in range(len(decoder_dims)-1):
            self.decoder.append(nn.Linear(decoder_dims[idx], decoder_dims[idx+1]))
            if idx+1 < len(decoder_dims)-1:
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
def run(cfg_path):

    batch_size = 8
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 50

    cngt_root = "/vol/tensusers5/jmartinez/datasets/cngt_single_signs_256"
    sb_root = "/vol/tensusers5/jmartinez/datasets/NGT_Signbank_256"
    sb_vocab_path = "/vol/tensusers5/jmartinez/datasets/signbank_vocab.gzip"
    model_root = "/vol/tensusers5/jmartinez/models/i3d"
    fig_output_root = "/vol/tensusers5/jmartinez/graphs"

    specific_glosses = ["GEBAREN-A", "JA-A"]
    run_name = "rq1"
    run_epochs = 50
    run_batch_size = 128
    run_lr = 0.1
    run_optimizer = "SGD"

    input_size = 256
    fold = "train"
    loading_mode = "balanced"
    window_size = 16
    clips_per_class = -1
    random_seed = 42
    use_cuda = True

    # get directory and filename for the checkpoints
    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    run_dir = f"{run_name}_{glosses_string}_{run_epochs}_{run_batch_size}_{run_lr}_{run_optimizer}"

    ckpt_files = [file for file in os.listdir(os.path.join(model_root, run_dir)) if file.endswith(".pt")]
    # take the last save checkpoint, which contains the minimum val loss
    ckpt_filename = ckpt_files[-1]
    # ckpt_filename = f"i3d_{str(ckpt_epoch).zfill(len(str(run_epochs)))}.pt"

    num_top_glosses = None

    sb_vocab = load_gzip(sb_vocab_path)
    gloss_to_id = sb_vocab['gloss_to_id']

    specific_gloss_ids = [gloss_to_id[gloss] for gloss in specific_glosses]

    cropped_input_size = input_size * 0.875

    test_transforms = transforms.Compose([transforms.CenterCrop(cropped_input_size)])

    print(f"Loading {fold} split...")
    dataset = I3Dataset(loading_mode=loading_mode,
                        cngt_root=cngt_root,
                        sb_root=sb_root,
                        sb_vocab_path=sb_vocab_path,
                        mode="rgb",
                        split=fold,
                        window_size=window_size,
                        transforms=test_transforms,
                        filter_num=num_top_glosses,
                        specific_gloss_ids=specific_gloss_ids,
                        clips_per_class=clips_per_class,
                        random_seed=random_seed)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    i3d = InceptionI3d(num_classes=len(dataset.class_encodings),
                       in_channels=3,
                       window_size=window_size,
                       input_size=cropped_input_size)

    if use_cuda:
        i3d.load_state_dict(torch.load(os.path.join(model_root, run_dir, ckpt_filename)))
    else:
        i3d.load_state_dict(torch.load(os.path.join(model_root, run_dir, ckpt_filename), map_location=torch.device('cpu')))

    if use_cuda:
        i3d.cuda()

    i3d.train(False)  # Set model to evaluate mode

    X_features = torch.zeros((1, 1024))

    print(f"Running datapoints through model...")
    with torch.no_grad():  # this deactivates gradient calculations, reducing memory consumption by A LOT
        with tqdm(dataloader, unit="batch") as tepoch:
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

    # this normalizes the features in order for them to have values in [0,1]
    train_features = X_features / X_features.amax(dim=1, keepdim=True)
    # train_features = nn.functional.normalize(train_features)

    train_dataloader = torch.utils.data.DataLoader(train_features, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    autoencoder = AutoEncoder(k=2)
    if use_cuda:
        autoencoder.cuda()

    # for param in autoencoder.parameters():
    #     print(param)
    #     break

    # summary(autoencoder, (1, 1024))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)

    outputs = []
    for epoch in tqdm(range(epochs)):
        for feature_vector in train_dataloader:
            optimizer.zero_grad()

            feature_vector = Variable(feature_vector.cuda())

            preds = autoencoder(feature_vector)
            loss = criterion(preds, feature_vector)

            loss.backward()
            optimizer.step()

        # print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        # outputs.append((epoch, preds, feature_vector))

    autoencoder.train(False)

    # MSE = []
    # for feature_vector in train_dataloader:
    #     feature_vector = Variable(feature_vector.cuda())
    #
    #     preds = autoencoder(feature_vector)
    #
    #     MSE.append(mean_squared_error(y_true=feature_vector.detach().cpu().numpy(), y_pred=preds.detach().cpu().numpy()))
    #
    # print(np.mean(MSE))

    all_MSE = []
    trimmed_autoencoder = autoencoder

    ks = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    for i in range(len(ks)):

        if i > 0:
            trimmed_autoencoder.encoder = nn.Sequential(*list(trimmed_autoencoder.encoder.children())[:-2])
            trimmed_autoencoder.decoder = nn.Sequential(*list(trimmed_autoencoder.decoder.children())[2:])

        # summary(trimmed_autoencoder, (1, 1024))

        MSE = []
        for feature_vector in train_dataloader:
            feature_vector = Variable(feature_vector.cuda())

            preds = autoencoder(feature_vector)

            MSE.append(mean_squared_error(y_true=feature_vector.detach().cpu().numpy(), y_pred=preds.detach().cpu().numpy()))

        all_MSE.append(np.mean(MSE))

    n_components = 2 ** np.arange(1, 10)

    plt.clf()

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    plt.plot(n_components, all_MSE, marker='o')
    # plt.plot(n_valid_components, my_mds_stress, marker='o', label='mds*')
    # plt.plot(n_valid_components, pca_stress, marker='o', label='pca*')
    #
    y_lims = plt.gca().get_ylim()
    y_range = np.abs(y_lims[0] - y_lims[1])

    for i, j in zip(n_components, all_MSE):
        plt.annotate(str(round(j, 2)), xy=(i + y_range * 0.05, j + y_range * 0.02))
    plt.xticks([2, 64, 128, 256, 512, 1024])
    plt.xlabel("Number of dimensions")
    plt.ylabel("Stress")
    plt.tight_layout()

    run_dir = run_dir.replace(":", ";")  # so that the files will work in Windows if a gloss has a ':' in it
    os.makedirs(fig_output_root, exist_ok=True)
    plt.savefig(os.path.join(fig_output_root, run_dir + '_autoencoder.png'))

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
