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

    # # THIS IS FOR PT-1hand:1
    # min_num_dims = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    # acc = [0.7368, 0.7985, 0.6045, 0.8077, 0.7206, 0.7042, 0.7303, 0.7770, 0.7737, 0.8511]
    # f1 = [0.7154, 0.8000, 0.5546, 0.8175, 0.7206, 0.7692, 0.7760, 0.7862, 0.8166, 0.8727]

    # THIS IS FOR WETEN-A
    min_num_dims = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    acc = [0.8045, 0.7970, 0.7333, 0.7259, 0.8148, 0.8248, 0.8311, 0.8841, 0.8248, 0.9366]
    f1 = [0.7903, 0.8235, 0.7188, 0.7176, 0.8252, 0.8154, 0.8227, 0.8961, 0.8500, 0.9434]

    colors = sns.color_palette('pastel')
    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Linguistic distance')
    ax1.set_ylabel("Min number of dimensions")
    ax1.tick_params(axis='y', color=colors[0], labelcolor=colors[0])
    ax1.set_yticks(n_dims)
    ax1.plot(ling_dist, min_num_dims, linestyle='-', marker='^', color=colors[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Performance")
    ax2.spines['right'].set_visible(True)
    ax2.set_xticks(np.arange(1, 11))
    ax2.plot(ling_dist, acc, linestyle='--', label='accuracy', marker='o', color=colors[1])
    ax2.plot(ling_dist, f1, linestyle='--', label='f1', marker='o', color=colors[2])
    ax2.legend(loc='best')

    ax2.set_axisbelow(True)  # grid lines are behind the rest
    ax2.yaxis.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
