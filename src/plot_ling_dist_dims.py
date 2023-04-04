import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys

def main():

    ling_dist = np.arange(1, 11)

    # # THIS IS FOR PT-1hand:1
    intrinsic_dims = [24.9, 22.5, 25.4, 25.9, 24.6, 23.1, 25.1, 25.9, 25.1, 23.6]
    acc = [0.7368, 0.7985, 0.6045, 0.8077, 0.7206, 0.7042, 0.7303, 0.7770, 0.7737, 0.8511]
    f1 = [0.7154, 0.8000, 0.5546, 0.8175, 0.7206, 0.7692, 0.7760, 0.7862, 0.8166, 0.8727]

    # THIS IS FOR WETEN-A
    # intrinsic_dims = [25.0, 22.3, 26.5, 23.9, 26.2, 23.9, 24.5, 25.1, 24.0, 23.2]
    # acc = [0.8045, 0.7970, 0.7333, 0.7259, 0.8148, 0.8248, 0.8311, 0.8841, 0.8248, 0.9366]
    # f1 = [0.7903, 0.8235, 0.7188, 0.7176, 0.8252, 0.8154, 0.8227, 0.8961, 0.8500, 0.9434]

    # THIS IS FOR DOOF-B
    # intrinsic_dims = [22, 24, 23, 26, 24, 22, 23, 20, 23, 22]
    # acc = [0.7984, 0.8468, 0.8548, 0.8615, 0.8110, 0.8030, 0.8521, 0.9124, 0.8992, 0.9323]
    # f1 = [0.8030, 0.8480, 0.8594,  0.8615, 0.8154, 0.7869, 0.8372, 0.9062, 0.9065, 0.9362]

    # THIS IS FOR DOOF-A
    # intrinsic_dims = [23.0, 20.1, 24.8, 23.5, 23.3, 21.5, 22.2, 23.9, 23.9, 21.0]
    # acc = [0.7901, 0.7742, 0.7024, 0.7439, 0.7528, 0.7582, 0.8111, 0.8817, 0.8132, 0.8842]
    # f1 = [0.7952, 0.7879, 0.6835, 0.7200, 0.7500, 0.7442, 0.7792, 0.8911, 0.8247, 0.8932]


    # THIS CODE BIT GETS THE INTRINSIC DIMENSIONS AND LINGUISTIC DISTANCE WRT ACC
    mapping_dict = {}
    for i, ac in enumerate(f1):
        mapping_dict[ac] = (intrinsic_dims[i], ling_dist[i])

    sorted_mapping = dict(sorted(mapping_dict.items()))

    sorted_f1 = list(sorted_mapping.keys())
    sorted_id = [value[0] for value in sorted_mapping.values()]
    sorted_ld = [value[1] for value in sorted_mapping.values()]

    sorted_f1_string = ['{:.3f}'.format(x) for x in sorted_f1]

    colors = sns.color_palette('pastel')
    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('F-score')
    ax1.set_ylabel("Linguistic distance")
    ax1.tick_params(axis='y', color=colors[0], labelcolor=colors[0])
    ax1.set_ylim([0, 11])
    ax1.plot(sorted_f1_string, sorted_ld, linestyle='-', marker='^', color=colors[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Intrinsic dimension")
    ax2.spines['right'].set_visible(True)
    ax2.plot(sorted_f1_string, sorted_id, linestyle='--', marker='o', color=colors[1])
    ax2.tick_params(axis='y', color=colors[1], labelcolor=colors[1])


    plt.tight_layout()
    plt.show()

    return

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Linguistic distance')
    ax1.set_ylabel("Intrinsic dimension")
    ax1.tick_params(axis='y', color=colors[0], labelcolor=colors[0])
    ax1.set_ylim([19, 27])
    ax1.plot(ling_dist, intrinsic_dims, linestyle='-', marker='^', color=colors[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Performance")
    ax2.spines['right'].set_visible(True)
    ax2.set_xticks(np.arange(1, 11))
    ax2.set_ylim([0.5, 1.0])
    ax2.set_yticks(np.linspace(0.5, 1.0, 6))
    ax2.plot(ling_dist, acc, linestyle='--', label='accuracy', marker='o', color=colors[1])
    ax2.plot(ling_dist, f1, linestyle='--', label='f1', marker='o', color=colors[2])
    ax2.legend(loc='upper left')

    # ax2.set_axisbelow(True)  # grid lines are behind the rest
    # ax2.yaxis.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
