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

from tqdm import tqdm
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.pytorch_i3d import InceptionI3d
from src.utils.i3d_dimensions_exp import InceptionI3d as InceptionDims
from src.utils.i3d_dimensions_conv import InceptionI3d as InceptionDimsConv
from src.utils.i3d_data import I3Dataset
from src.utils import spatial_transforms


def main(params):
    root = params.root
    cngt_folder = params.cngt_folder
    sb_folder = params.sb_folder
    sb_vocab_file = params.sb_vocab_file
    fig_output_root = params.fig_output_root
    specific_glosses = params.specific_glosses
    window_size = params.window_size
    batch_size = params.batch_size
    loading_mode = params.loading_mode

    cngt_zip = os.path.join(root, cngt_folder + ".zip")
    sb_zip = os.path.join(root, sb_folder + ".zip")
    sb_vocab_path = os.path.join(root, sb_vocab_file)

    # get glosses from the class encodings
    sb_vocab = load_gzip(sb_vocab_path)
    gloss_to_id = sb_vocab['gloss_to_id']

    specific_glosses = specific_glosses.split(",")
    specific_gloss_ids = [gloss_to_id[gloss] for gloss in list(specific_glosses)]

    print("Loading training split...")
    train_dataset = I3Dataset(loading_mode=loading_mode,
                              cngt_zip=cngt_zip,
                              sb_zip=sb_zip,
                              sb_vocab_path=sb_vocab_path,
                              mode="rgb",
                              split="train",
                              window_size=window_size,
                              transforms=None,
                              filter_num=None,
                              specific_gloss_ids=specific_gloss_ids,
                              diagonal_videos_path=None)

    print("Loading validation split...")
    val_dataset = I3Dataset(loading_mode="balanced",
                            cngt_zip=cngt_zip,
                            sb_zip=sb_zip,
                            sb_vocab_path=sb_vocab_path,
                            mode="rgb",
                            split="val",
                            window_size=window_size,
                            transforms=None,
                            filter_num=None,
                            specific_gloss_ids=specific_gloss_ids,
                            diagonal_videos_path=None)

    print("Loading test split...")
    test_dataset = I3Dataset(loading_mode="balanced",
                             cngt_zip=cngt_zip,
                             sb_zip=sb_zip,
                             sb_vocab_path=sb_vocab_path,
                             mode="rgb",
                             split="test",
                             window_size=window_size,
                             transforms=None,
                             filter_num=None,
                             specific_gloss_ids=specific_gloss_ids,
                             diagonal_videos_path=None)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    dataloaders = [train_dataloader, val_dataloader, test_dataloader]

    clip_duration_per_split = []
    for dataloader in dataloaders:
        with tqdm(dataloader, unit="batch") as tepoch:

            clip_durations = []
            for data in tepoch:
                _, _, video_paths = data
                for video_path in video_paths:
                    # filter out SB videos
                    if "cngt" in video_path.split("/")[-2]:
                        clip = os.path.basename(video_path)
                        start_ms = int(clip.split("_")[4])
                        end_ms = int(clip.split("_")[5])
                        n_frames = math.ceil(25 * (end_ms - start_ms) / 1000)
                        clip_durations.append(n_frames)
        clip_duration_per_split.append(clip_durations)

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    upper_quartile = np.percentile(clip_duration_per_split[0], 75)
    lower_quartile = np.percentile(clip_duration_per_split[0], 25)
    iqr = upper_quartile - lower_quartile

    clip_duration_per_split[0] = np.array(clip_duration_per_split[0])

    upper_whisker = clip_duration_per_split[0][np.where(clip_duration_per_split[0] <= upper_quartile + 1.5 * iqr, True, False)].max()
    lower_whisker = clip_duration_per_split[0][np.where(clip_duration_per_split[0] >= lower_quartile - 1.5 * iqr, True, False)].min()

    plt.hist(clip_duration_per_split[0], bins='auto', align='mid')
    plt.hist(clip_duration_per_split[1], bins='auto', align='mid')
    plt.hist(clip_duration_per_split[2], bins='auto', align='mid')
    plt.xlim([lower_whisker - 1, upper_whisker + 1])
    plt.tight_layout()

    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"

    os.makedirs(fig_output_root, exist_ok=True)
    plt.savefig(os.path.join(fig_output_root, f"{glosses_string}_{loading_mode}_frames_per_split_distribution.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
    )

    parser.add_argument(
        "--cngt_folder",
        type=str,
    )

    parser.add_argument(
        "--sb_folder",
        type=str,
    )

    parser.add_argument(
        "--sb_vocab_file",
        type=str,
    )

    parser.add_argument(
        "--fig_output_root",
        type=str,
    )

    parser.add_argument(
        "--specific_glosses",
        type=str,
    )

    parser.add_argument(
        "--window_size",
        type=int,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
    )

    parser.add_argument(
        "--loading_mode",
        type=str,
    )

    params, _ = parser.parse_known_args()
    main(params)
