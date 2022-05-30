import os
import sys
sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from src.utils import videotransforms
from src.utils.i3d_data import I3Dataset
from src.utils.helpers import load_config
from src.utils.pytorch_i3d import InceptionI3d
from src.utils.loss import f1_loss

def test(cfg_path, checkpoint_filename, mode="rgb"):
    cfg = load_config(cfg_path)
    training_cfg = cfg.get("training")

    batch_size = training_cfg.get("batch_size")
    save_model = training_cfg.get("model_dir")
    weights_dir = training_cfg.get("weights_dir")

    # data configs
    cngt_zip = cfg.get("data").get("cngt_clips_path")
    sb_zip = cfg.get("data").get("signbank_path")
    window_size = cfg.get("data").get("window_size")

    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    print("Loading test split...")
    test_dataset = I3Dataset(cngt_zip, sb_zip, mode, "test", window_size, test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True)

    # set up the model
    i3d = InceptionI3d(len(test_dataset.class_encodings))
    i3d.load_state_dict(torch.load(os.path.join(weights_dir, checkpoint_filename)))
    i3d.cuda()

    for data in tqdm(test_dataloader):
        i3d.train(False)

        inputs, labels = data

        inputs = inputs.cuda()
        labels = labels.cuda()

        preds = i3d(inputs)
        preds = F.upsample(preds, inputs.size(2), mode='linear')
        preds = torch.nn.Softmax(dim=1)(preds


def main(params):
    config_path = params.config_path
    checkpoint_filename = params.checkpoint_filename
    test(config_path, checkpoint_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
    )

    parser.add_argument(
        "--checkpoint_filename",
        type=str,
    )

    params, _ = parser.parse_known_args()
    main(params)