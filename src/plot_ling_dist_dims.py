import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys

def main():

    ling_dist = np.arange(1, 11)

    # # THIS IS FOR PT-1hand:1
    # mle_id = [24.84, 24.34, 23.51, 21.94, 23.67, 24.72, 23.12, 21.72, 23.76, 24.89]
    # twonn_id = [18.41, 19.57, 18.65, 17.64, 17.59, 19.64, 16.85, 16.56, 14.45, 18.27]
    # acc = [0.7368, 0.7985, 0.6045, 0.8077, 0.7206, 0.7042, 0.7303, 0.7770, 0.7737, 0.8511]
    # f1 = [0.7154, 0.8000, 0.5546, 0.8175, 0.7206, 0.7692, 0.7760, 0.7862, 0.8166, 0.8727]

    # THIS IS FOR WETEN-A
    # mle_id = [24.28, 20.72, 25.72, 22.27, 24.0, 22.74, 22.97, 23.41, 23.05, 22.52]
    # twonn_id = [19.52, 19.83, 22.39, 18.77, 19.35, 15.03, 16.23, 17.66, 18.36, 18.5]
    # acc = [0.8045, 0.7970, 0.7333, 0.7259, 0.8148, 0.8248, 0.8311, 0.8841, 0.8248, 0.9366]
    # f1 = [0.7903, 0.8235, 0.7188, 0.7176, 0.8252, 0.8154, 0.8227, 0.8961, 0.8500, 0.9434]

    # THIS IS FOR DOOF-B
    # mle_id = [21.46, 23.16, 21.39, 23.31, 23.27, 19.95, 20.98, 18.66, 21.71, 19.98]
    # twonn_id = [15.51, 20.35, 19.54, 20.63, 15.78, 15.1, 16.17, 16.83, 16.21, 16.41]
    # acc = [0.7984, 0.8468, 0.8548, 0.8615, 0.8110, 0.8030, 0.8521, 0.9124, 0.8992, 0.9323]
    # f1 = [0.8030, 0.8480, 0.8594,  0.8615, 0.8154, 0.7869, 0.8372, 0.9062, 0.9065, 0.9362]

    # THIS IS FOR DOOF-A
    mle_id = [23.16, 18.2, 23.31, 21.42, 22.47, 19.57, 21.37, 22.7, 22.34, 20.99]
    twonn_id = [19.7, 12.78, 16.18, 15.84, 15.81, 11.64, 13.5, 15.46, 14.63, 13.54]
    acc = [0.7901, 0.7742, 0.7024, 0.7439, 0.7528, 0.7582, 0.8111, 0.8817, 0.8132, 0.8842]
    f1 = [0.7952, 0.7879, 0.6835, 0.7200, 0.7500, 0.7442, 0.7792, 0.8911, 0.8247, 0.8932]


    # THIS CODE BIT GETS THE INTRINSIC DIMENSIONS AND LINGUISTIC DISTANCE WRT ACC
    # mapping_dict = {}
    # for i, ac in enumerate(f1):
    #     mapping_dict[ac] = (intrinsic_dims[i], ling_dist[i])
    #
    # sorted_mapping = dict(sorted(mapping_dict.items()))
    #
    # sorted_f1 = list(sorted_mapping.keys())
    # sorted_id = [value[0] for value in sorted_mapping.values()]
    # sorted_ld = [value[1] for value in sorted_mapping.values()]
    #
    # sorted_f1_string = ['{:.3f}'.format(x) for x in sorted_f1]
    #
    # colors = sns.color_palette('pastel')
    # plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")
    #
    # fig, ax1 = plt.subplots()
    #
    # ax1.set_xlabel('F-score')
    # ax1.set_ylabel("Linguistic distance")
    # ax1.tick_params(axis='y', color=colors[0], labelcolor=colors[0])
    # ax1.set_ylim([0, 11])
    # ax1.plot(sorted_f1_string, sorted_ld, linestyle='-', marker='^', color=colors[0])
    #
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("Intrinsic dimension")
    # ax2.spines['right'].set_visible(True)
    # ax2.plot(sorted_f1_string, sorted_id, linestyle='--', marker='o', color=colors[1])
    # ax2.tick_params(axis='y', color=colors[1], labelcolor=colors[1])
    #
    #
    # plt.tight_layout()
    # plt.show()
    #
    # return

    colors = sns.color_palette('pastel')
    plt.style.use(Path(__file__).parent.resolve() / "../plot_style.txt")

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Linguistic distance')
    ax1.set_ylabel("Intrinsic dimension")
    # ax1.tick_params(axis='y', color=colors[0], labelcolor=colors[0])
    # ax1.set_ylim([19, 27])
    ax1.plot(ling_dist, mle_id, linestyle='-', label='MLE', marker='^', color=colors[0])
    ax1.plot(ling_dist, twonn_id, linestyle='-', label='TwoNN', marker='^', color=colors[1])
    ax1.legend(loc='lower left')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Performance")
    ax2.spines['right'].set_visible(True)
    ax2.set_xticks(np.arange(1, 11))
    ax2.set_ylim([0.5, 1.0])
    ax2.set_yticks(np.linspace(0.5, 1.0, 6))
    ax2.plot(ling_dist, acc, linestyle='--', label='accuracy', marker='o', color=colors[2])
    ax2.plot(ling_dist, f1, linestyle='--', label='f1', marker='o', color=colors[3])
    ax2.legend(loc='lower right')

    # ax2.set_axisbelow(True)  # grid lines are behind the rest
    # ax2.yaxis.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
