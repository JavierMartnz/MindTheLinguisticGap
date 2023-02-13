import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

from src.utils.helpers import load_config


def main(params):
    trained_models_parent = params.trained_models_parent
    fig_output_root = params.fig_output_root
    config_path = params.config_path

    cfg = load_config(config_path)
    training_cfg = cfg.get("training")

    # training configs
    specific_glosses = training_cfg.get("specific_glosses")
    run_name = training_cfg.get("run_name")
    epochs = training_cfg.get("epochs")
    init_lr = training_cfg.get("init_lr")
    batch_size = training_cfg.get("batch_size")

    optimizer = 'SGD'

    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    weights_root = f"{run_name}_{glosses_string}_{epochs}_{batch_size}_{init_lr}_{optimizer}"

    training_file_path = os.path.join(trained_models_parent, weights_root, 'training_history.txt')

    with open(training_file_path, 'r') as file:
        training_history = json.load(file)

    epochs = np.arange(1, len(training_history['train_loss']) + 1)

    colors = sns.color_palette('pastel')

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    # plt.plot(epochs, training_history['train_accuracy'], label='train_acc', linestyle='--')
    plt.plot(epochs, training_history['train_f1'], label='train_f1', linestyle='--')
    # plt.plot(epochs, training_history['val_accuracy'], label='val_acc', linestyle='-')
    plt.plot(epochs, training_history['val_f1'], label='val_f1', linestyle='-')
    plt.legend()
    plt.tight_layout()
    # plt.show()

    os.makedirs(fig_output_root, exist_ok=True)
    plt.savefig(os.path.join(fig_output_root, os.path.basename(trained_model_root) + '_metrics.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trained_models_parent",
        type=str,
        default="D:/Thesis/models/i3d"
    )

    parser.add_argument(
        "--fig_output_root",
        type=str,
        default="D:/Thesis/graphs"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="C:/Users/Javi/PycharmProjects/MindTheLinguisticGap/config_local.yaml"
    )

    params, _ = parser.parse_known_args()
    main(params)
