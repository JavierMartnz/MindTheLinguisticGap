import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys

def main():

    ling_dist = np.arange(1, 11)

    # # THIS IS FOR PT-1hand:1
    mle_id = [24.84, 24.34, 23.51, 21.94, 23.67, 24.72, 23.12, 21.72, 23.76, 24.89]
    twonn_id = [18.41, 19.57, 18.65, 17.64, 17.59, 19.64, 16.85, 16.56, 14.45, 18.27]
    acc = [0.7440, 0.8320, 0.6148, 0.8110, 0.7381, 0.6667, 0.7266, 0.7953, 0.7583, 0.8425]
    f1 = [0.7193, 0.8346, 0.5524, 0.8235, 0.7360, 0.7153, 0.7654, 0.8000, 0.7883, 0.8592]

    # THIS IS FOR WETEN-A
    # mle_id = [24.28, 20.72, 25.72, 22.27, 24.0, 22.74, 22.97, 23.41, 23.05, 22.52]
    # twonn_id = [19.52, 19.83, 22.39, 18.77, 19.35, 15.03, 16.23, 17.66, 18.36, 18.5]
    # acc = [0.7953, 0.7823, 0.7891, 0.7008, 0.8319, 0.8125, 0.9160, 0.9426, 0.8889, 0.9756]
    # f1 = [0.7833, 0.8000, 0.7568, 0.7164, 0.8182, 0.8125, 0.9147, 0.9412, 0.8955, 0.9760]

    # THIS IS FOR DOOF-B
    # mle_id = [21.46, 23.16, 21.39, 23.31, 23.27, 19.95, 20.98, 18.66, 21.71, 19.98]
    # twonn_id = [15.51, 20.35, 19.54, 20.63, 15.78, 15.1, 16.17, 16.83, 16.21, 16.41]
    # acc = [0.8496, 0.8898, 0.8696, 0.8393, 0.8390, 0.8583, 0.8974, 0.9417, 0.9339, 0.9344]
    # f1 = [0.8496, 0.8889, 0.8739, 0.8525, 0.8348, 0.8595, 0.9000, 0.9381, 0.9333, 0.9344]

    # THIS IS FOR DOOF-A
    # mle_id = [23.16, 18.2, 23.31, 21.42, 22.47, 19.57, 21.37, 22.7, 22.34, 20.99]
    # twonn_id = [19.7, 12.78, 16.18, 15.84, 15.81, 11.64, 13.5, 15.46, 14.63, 13.54]
    # acc = [0.8875, 0.9079, 0.7500, 0.7200, 0.7857, 0.8571, 0.8961, 0.8657, 0.8684, 0.9067]
    # f1 = [0.8861, 0.9157, 0.7397, 0.7200, 0.7619, 0.8675, 0.9000, 0.8615, 0.8810, 0.9136]


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
    plt.style.use(Path(__file__).parent.resolve() / "../../plot_style.txt")

    # plt.xlabel("Linguistic distance")
    # plt.ylabel("Performance")
    # plt.plot(ling_dist, acc, label="accuracy", marker='o', color=colors[0])
    # plt.plot(ling_dist, f1, label="f-score", marker='o', color=colors[1])
    # plt.ylim([0.5, 1.0])
    # plt.xticks(np.arange(1, 11))
    # plt.legend(loc='best')
    # plt.grid(axis='y', alpha=0.3)

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
    # ax2.set_ylim([0.5, 1.0])
    # ax2.set_yticks(np.linspace(0.5, 1.0, 6))
    ax2.plot(ling_dist, acc, linestyle='--', label='accuracy', marker='o', color=colors[2])
    ax2.plot(ling_dist, f1, linestyle='--', label='f1', marker='o', color=colors[3])
    ax2.legend(loc='lower right')

    ax2.set_axisbelow(True)  # grid lines are behind the rest
    ax2.yaxis.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
