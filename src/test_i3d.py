import os
import sys
sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


from src.utils import videotransforms
from src.utils.i3d_data import I3Dataset
from src.utils.helpers import load_config
from src.utils.pytorch_i3d import InceptionI3d
from src.utils.loss import f1_loss

def test(cfg_path, mode="rgb"):
    cfg = load_config(cfg_path)
    test_cfg = cfg.get("test")

    # test parameters
    save_model = test_cfg.get("model_dir")
    run_name = test_cfg.get("run_name")
    run_batch_size = test_cfg.get("run_batch_size")
    optimizer = test_cfg.get("optimizer").upper()
    learning_rate = test_cfg.get("lr")
    num_epochs = test_cfg.get("epochs")
    ckpt_epoch = test_cfg.get("ckpt_epoch")
    ckpt_step = test_cfg.get("ckpt_step")
    batch_size = test_cfg.get("batch_size")

    # data configs
    cngt_zip = cfg.get("data").get("cngt_clips_path")
    sb_zip = cfg.get("data").get("signbank_path")
    window_size = cfg.get("data").get("window_size")

    num_top_glosses = 2

    print("Loading test split...")
    dataset = I3Dataset(cngt_zip, sb_zip, mode, 'test', window_size, transforms=None, filter_num=num_top_glosses)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    run_dir = f"b{run_batch_size}_{optimizer}_lr{learning_rate}_ep{num_epochs}_{run_name}"
    ckpt_filename = f"i3d_{ckpt_epoch}_{ckpt_step}.pt"

    # load model and specified checkpoint
    i3d = InceptionI3d(num_classes=len(dataset.class_encodings), in_channels=3, window_size=16)
    i3d.load_state_dict(torch.load(os.path.join(save_model, run_dir, ckpt_filename)))

    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    total_pred = []
    total_true = []

    with torch.no_grad():  # this deactivates gradient calculations, reducing memory consumption by A LOT
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                # get the inputs
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # forward pass of the inputs through the network
                per_frame_logits = i3d(inputs)

                # upsample output to input size
                per_frame_logits = F.interpolate(per_frame_logits, size=inputs.size(2), mode='linear')

                # get prediction
                y_pred = np.argmax(per_frame_logits.detach().cpu().numpy(), axis=1)
                y_true = np.argmax(labels.detach().cpu().numpy(), axis=1)

                if len(total_pred) == 0:
                    total_pred = y_pred
                    total_true = y_true
                else:
                    total_pred.append(y_pred, axis=0)
                    total_true.append(y_true, axis=0)

                # calculate batch metrics by averaging
                # batch_acc = np.mean([accuracy_score(y_true[i], y_pred[i]) for i in range(np.shape(y_pred)[0])])
                # batch_f1 = np.mean([f1_score(y_true[i], y_pred[i], average='macro') for i in range(np.shape(y_pred)[0])])
                #
                # acc_list.append(batch_acc)
                # f1_list.append(batch_f1)

    print(np.shape(total_pred))


def main(params):
    config_path = params.config_path
    test(config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
    )

    params, _ = parser.parse_known_args()
    main(params)