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
    min_num_dims = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    performance = [0.7, 0.8, 0.8, 0.7, 0.8, 0.9, 0.7, 0.8, 0.85, 0.92]

    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    fig, ax1 = plt.subplots()
    ax1.plot(ling_dist, min_num_dims, linestyle='-', marker='o')
    ax1.set_xlabel("Pair-wise linguistic distance")
    ax1.set_ylabel("Min number of dimensions")

    ax2 = ax1.twinx()
    ax2.plot(ling_dist, performance, linestyle='--', marker='o')
    ax2.set_ylabel("Performance")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
