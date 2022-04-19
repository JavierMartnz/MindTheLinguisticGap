import os
import shutil

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import argparse

from src.utils.my_collate import my_collate
from src.utils.helpers import load_config, set_seed
from src.utils.pretrain_data import load_data, get_class_encodings
from src.utils.i3dpt import I3D


def train(cfg_path: str) -> None:
    assert os.path.isfile(cfg_path), f"{cfg_path} is not a config file"
    cfg = load_config(cfg_path)
    training_cfg = cfg["training"]
    set_seed(seed=training_cfg.get("random_seed", 42))

    train_dataset, val_dataset = load_data(cfg.get("data"), ['train', 'val'])

    print("Creating training Dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=training_cfg.get("batch_size"), shuffle=True, num_workers=0,
                                  pin_memory=True, collate_fn=my_collate)
    print("Creating validation Dataloader...")
    val_dataloader = DataLoader(val_dataset, batch_size=training_cfg.get("batch_size"), shuffle=True, num_workers=0,
                                pin_memory=True, collate_fn=my_collate)

    num_classes = len(get_class_encodings(cfg.get("data").get("cngt_clips_path"), cfg.get("data").get("signbank_path")))

    # model = InceptionI3d(num_classes=num_classes,
    #                      spatial_squeeze=True,
    #                      final_endpoint='Predictions',
    #                      name='inception_i3d',
    #                      in_channels=3,
    #                      dropout_keep_prob=0.5
    # )

    model = I3D(num_classes=num_classes,
                modality='rgb',
                dropout_prob=0.5,
                name='i3d')

    model = torch.nn.DataParallel(model).cuda()

    lr = training_cfg.get("init_lr")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    epochs = training_cfg.get("epochs")
    min_val_loss = np.inf

    print(
        f"Training parameters summary:\n\t-batch size={training_cfg.get('batch_size')}\n\t-initial learning rate={lr}")

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        cnt = 0
        for data in train_dataloader:
            cnt += 1
            optimizer.zero_grad()
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs.float())
            loss = F.binary_cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                pred = model(inputs.float())
                loss = F.binary_cross_entropy(pred, labels)
                val_loss += loss.item()

        lr_sched.step(val_loss)  # this always after the optimizer step

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)

        print(f"Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs} -- Training loss: {train_loss:.4f} \t\t "
              f"Validation loss: {val_loss:.4f}")
        if val_loss < min_val_loss:
            print(f"\tValidation loss improved from {min_val_loss:.4f} to {val_loss:.4f}... Saving model")
            min_val_loss = val_loss
            torch.save(model.module.state_dict(), training_cfg.get("save_model") + str(epoch).zfill(6) + '.pt')

    # once the training is done, remove unzipped folders
    shutil.rmtree(cfg.get("data").get("cngt_clips_path")[:-4])
    shutil.rmtree(cfg.get("data").get("signbank_path")[:-4])

def main(params):

    config_path = params.config_path
    train(config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
    )
    params, _ = parser.parse_known_args()
    main(params)
