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
    n_dims = 2 ** np.arange(6, 11)
    min_num_dims = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    acc = [0.6824, 0.7326, 0.5357, 0.7683, 0.6049, 0.7849, 0.6837, 0.8222, 0.7500, 0.9130]
    f1 = [0.7033, 0.7677, 0.5063, 0.7467, 0.6596, 0.8113, 0.6931, 0.8298, 0.7800, 0.9259]

    colors = sns.color_palette('pastel')
    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Linguistic distance')
    ax1.set_ylabel("Min number of dimensions")
    ax1.plot(ling_dist, min_num_dims, linestyle='-', marker='o', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.set_yticks(n_dims)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Performance")
    ax2.spines['right'].set_visible(True)
    ax2.plot(ling_dist, acc, linestyle='--', label='accuracy', marker='o')
    ax2.plot(ling_dist, f1, linestyle='--', label='f1', marker='o')
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
