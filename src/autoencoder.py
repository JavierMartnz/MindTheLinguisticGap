import os
import sys

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

from src.utils import videotransforms
from src.utils.helpers import load_config
from src.utils.util import make_dir, load_gzip

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

# why an autoencoder is similar to PCA
# https://towardsdatascience.com/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-i-1f01f821999b

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

    # n_layers = 8

    train_features = load_gzip("D:/Thesis/datasets/i3d_features.gzip")

    # this normalizes the features in order for them to have values in [0,1]
    train_features = train_features / train_features.amax(dim=1, keepdim=True)
    # train_features = nn.functional.normalize(train_features)

    train_dataloader = torch.utils.data.DataLoader(train_features, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    autoencoder = AutoEncoder(k=2)
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

    MSE = []
    for feature_vector in train_dataloader:
        feature_vector = Variable(feature_vector.cuda())

        preds = autoencoder(feature_vector)

        MSE.append(mean_squared_error(y_true=feature_vector.detach().cpu().numpy(), y_pred=preds.detach().cpu().numpy()))

    print(np.mean(MSE))


    # trimmed_autoencoder = autoencoder
    #
    # ks = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    #
    # for i in range(len(ks)):
    #
    #     if i > 0:
    #         trimmed_autoencoder.encoder = nn.Sequential(*list(trimmed_autoencoder.encoder.children())[:-2])
    #         trimmed_autoencoder.decoder = nn.Sequential(*list(trimmed_autoencoder.decoder.children())[2:])
    #
    #     # summary(trimmed_autoencoder, (1, 1024))
    #
    #     MSE = []
    #     for feature_vector in train_dataloader:
    #         feature_vector = Variable(feature_vector.cuda())
    #
    #         preds = autoencoder(feature_vector)
    #
    #         MSE.append(mean_squared_error(y_true=feature_vector.detach().cpu().numpy(), y_pred=preds.detach().cpu().numpy()))
    #
    #     print(np.mean(MSE))

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
