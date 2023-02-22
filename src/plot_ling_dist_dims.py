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


def main():

    ling_dist = np.arange(1, 11)
    min_num_dims = [128, ]
    performance = []

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    plt.plot(ling_dist, min_num_dims, label='dimesnison', linestyle='--')
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

    os.makedirs(fig_output_root, exist_ok=True)
    plt.savefig(os.path.join(fig_output_root, weights_root + '_loss.png'))


if __name__ == "__main__":
    main()
