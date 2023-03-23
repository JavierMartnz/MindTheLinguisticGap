import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys

def main():

    ling_dist = np.arange(1, 11)
    n_dims = 2 ** np.arange(6, 11)

    # # THIS IS FOR PT-1hand:1
    min_num_dims = np.ones(10) * 256
    # acc = [0.7368, 0.7985, 0.6045, 0.8077, 0.7206, 0.7042, 0.7303, 0.7770, 0.7737, 0.8511]
    # f1 = [0.7154, 0.8000, 0.5546, 0.8175, 0.7206, 0.7692, 0.7760, 0.7862, 0.8166, 0.8727]

    # THIS IS FOR WETEN-A
    # min_num_dims = np.ones(10) * 64
    # acc = [0.8045, 0.7970, 0.7333, 0.7259, 0.8148, 0.8248, 0.8311, 0.8841, 0.8248, 0.9366]
    # f1 = [0.7903, 0.8235, 0.7188, 0.7176, 0.8252, 0.8154, 0.8227, 0.8961, 0.8500, 0.9434]

    # THIS IS FOR DOOF-B
    id = [26, 38, 32, 31, 36, 21, 25, 24, 37, 30]
    acc = [0.7984, 0.8468, 0.8548, 0.8615, 0.8110, 0.8030, 0.8521, 0.9124, 0.8992, 0.9323]
    f1 = [0.8030, 0.8480, 0.8594,  0.8615, 0.8154, 0.7869, 0.8372, 0.9062, 0.9065, 0.9362]

    # THIS IS FOR DOOF-A
    # min_num_dims = np.ones(10) * 64
    # acc = [0.7901, 0.7742, 0.7024, 0.7439, 0.7528, 0.7582, 0.8111, 0.8817, 0.8132, 0.8842]
    # f1 = [0.7952, 0.7879, 0.6835, 0.7200, 0.7500, 0.7442, 0.7792, 0.8911, 0.8247, 0.8932]

    colors = sns.color_palette('pastel')
    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Linguistic distance')
    ax1.set_ylabel("Min number of dimensions")
    ax1.tick_params(axis='y', color=colors[0], labelcolor=colors[0])
    ax1.plot(ling_dist, id, linestyle='-', marker='^', color=colors[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Performance")
    ax2.spines['right'].set_visible(True)
    ax2.set_xticks(np.arange(1, 11))
    ax2.set_ylim([0.5, 1.0])
    ax2.set_yticks(np.linspace(0.5, 1.0, 6))
    ax2.plot(ling_dist, acc, linestyle='--', label='accuracy', marker='o', color=colors[1])
    ax2.plot(ling_dist, f1, linestyle='--', label='f1', marker='o', color=colors[2])
    ax2.legend(loc='upper left')

    ax2.set_axisbelow(True)  # grid lines are behind the rest
    ax2.yaxis.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
