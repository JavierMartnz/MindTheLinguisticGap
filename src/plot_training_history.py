import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
import sys

sys.path.append("/vol/tensusers5/jmartinez/MindTheLinguisticGap")

from src.utils.helpers import load_config


def plot_train_history(specific_glosses: list, config: dict, trained_models_root: str, fig_output_root: str):
    run_name = config.get("run_name")
    run_epochs = config.get("run_epochs")
    run_lr = config.get("run_lr")
    run_batch_size = config.get("run_batch_size")
    run_optimizer = config.get("run_optimizer")

    glosses_string = f"{specific_glosses[0]}_{specific_glosses[1]}"
    weights_root = f"{run_name}_{glosses_string}_{run_epochs}_{run_batch_size}_{run_lr}_{run_optimizer}"

    training_file_path = os.path.join(trained_models_root, weights_root, 'training_history.txt')

    with open(training_file_path, 'r') as file:
        training_history = json.load(file)

    # do this if files have to be open on Windows later
    weights_root = weights_root.replace(':', ';')

    epochs = np.arange(1, len(training_history['train_loss']) + 1)

    colors = sns.color_palette('pastel')

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    plt.plot(epochs, training_history['train_accuracy'], label='train_acc', linestyle='--')
    plt.plot(epochs, training_history['train_f1'], label='train_f1', linestyle='--')
    plt.plot(epochs, training_history['val_accuracy'], label='val_acc', linestyle='-')
    plt.plot(epochs, training_history['val_f1'], label='val_f1', linestyle='-')
    plt.legend()
    plt.tight_layout()
    # plt.show()

    os.makedirs(fig_output_root, exist_ok=True)
    plt.savefig(os.path.join(fig_output_root, weights_root + '_metrics.png'))

    plt.clf()

    plt.plot(epochs, training_history['train_loss'], label='train_loss', linestyle='--')
    plt.plot(epochs, training_history['val_loss'], label='val_loss', linestyle='-')
    plt.legend()
    plt.tight_layout()
    # plt.show()

    print(f"The epoch with the min loss was {training_history['val_loss'].index(min(training_history['val_loss'])) + 1}")

    os.makedirs(fig_output_root, exist_ok=True)
    plt.savefig(os.path.join(fig_output_root, weights_root + '_loss.png'))


def main(params):
    trained_models_root = params.trained_models_root
    fig_output_root = params.fig_output_root
    config_path = params.config_path

    config = load_config(config_path)

    # training configs
    reference_sign = config.get("reference_sign")
    test_signs = config.get("test_signs")

    for sign in test_signs:
        plot_train_history([reference_sign, sign], config, trained_models_root, fig_output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trained_models_root",
        type=str,
    )

    parser.add_argument(
        "--fig_output_root",
        type=str,
    )

    parser.add_argument(
        "--config_path",
        type=str,
    )

    params, _ = parser.parse_known_args()
    main(params)
