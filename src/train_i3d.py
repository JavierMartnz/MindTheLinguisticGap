import os
import shutil
import sys

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")
import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.autograd import Variable
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
from time import sleep

from src.utils.my_collate import my_collate
from src.utils.helpers import load_config, set_seed
from src.utils.pretrain_data import load_data, get_class_encodings
from src.utils.i3dpt import I3D


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    # f1.requires_grad = is_training
    return f1.item()


class TrainManager:
    def __init__(self, model: torch.nn.Module, config: dict) -> None:
        train_config = config.get("training")
        data_config = config.get("data")
        self.train_config = train_config
        self.data_config = data_config
        # set parameters from config files
        self.use_cuda = train_config.get("use_cuda")  # this before building optimizer
        self.window_size = data_config.get("window_size")
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.model = model
        if self.use_cuda:
            self.model.cuda()
        self.optimizer = optim.SGD(model.parameters(), lr=train_config.get("init_lr"),
                                   momentum=train_config.get("momentum"), weight_decay=train_config.get("weight_decay"))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.batch_size = train_config.get("batch_size")
        self.epochs = train_config.get("epochs")
        self.model_dir = train_config.get("model_dir")
        # initialize training statistics
        self.steps = 0
        self.epoch = 0
        self.best_loss = np.inf
        self.train_loader = None
        self.val_loader = None

    def do_epoch(self, set_name):
        assert set_name in {'train', 'val'}
        acc_loss = 0
        loader = None
        if set_name == 'train':
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        with tqdm(loader, unit="batch") as tloader:
            for batch in tloader:
                tloader.set_description(f"Epoch {str(self.epoch + 1).zfill(len(str(self.epochs)))}/{self.epochs}")
                inputs, labels = batch
                if self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels)
                acc_loss += loss.item()
                if set_name == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.steps += 1

                avg_f1 = np.mean([f1_loss(y_true=labels[b], y_pred=outputs[b]) for b in range(labels.size(0))])
                tloader.set_postfix(loss=loss.item(), f1=avg_f1)
                sleep(0.1)

        return acc_loss

    def train_and_validate(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        print("Creating training Dataloader...")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                       pin_memory=True)
        print("Creating validation Dataloader...")
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                     pin_memory=True)

        while self.epoch < self.epochs:
            # print(f"Epoch {str(self.epoch + 1).zfill(len(str(self.epochs)))}/{self.epochs} --", end=" ")
            train_loss = self.do_epoch('train')
            with torch.no_grad():
                val_loss = self.do_epoch('val')

            if val_loss < self.best_loss:  # this is for minimizing metrics
                print(f"\tValidation loss improved from {self.best_loss:.4f} to {val_loss:.4f}")
                self.best_loss = val_loss
                print("Saving checkpoint")
                torch.save(self.model.module.state_dict(), self.model_dir + str(self.epoch).zfill(6) + '.pt')

            if self.scheduler is not None:
                self.scheduler.step(val_loss)  # after optimizer step when using ReduceLROnPlateau

        # once the training is done, remove unzipped folders
        # shutil.rmtree(self.data_config.get("cngt_clips_path")[:-4])
        # shutil.rmtree(self.data_config.get("signbank_path")[:-4])


def train(cfg_path: str) -> None:
    assert os.path.isfile(cfg_path), f"{cfg_path} is not a config file"
    cfg = load_config(cfg_path)
    training_cfg = cfg.get("training")
    set_seed(seed=training_cfg.get("random_seed", 42))

    train_dataset, val_dataset = load_data(cfg.get("data"), ['train', 'val'])
    num_classes = len(get_class_encodings(cfg.get("data").get("cngt_clips_path"), cfg.get("data").get("signbank_path")))

    model = I3D(num_classes=num_classes,
                modality='rgb',
                dropout_prob=0.5,
                name='i3d')

    model = torch.nn.DataParallel(model).cuda()
    # summary(model, (3, 64, 256, 256))

    trainer = TrainManager(model=model, config=cfg)
    trainer.train_and_validate(train_dataset, val_dataset)


def main(params):

    max_mem = 0
    best_gpu_idx = None
    for d in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(d).total_memory
        if mem > max_mem:
            max_mem = mem
            best_gpu_idx = d

    print(f"Running code in {torch.cuda.get_device_name(d)}")
    with torch.cuda.device(best_gpu_idx):
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
