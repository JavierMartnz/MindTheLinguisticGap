import matplotlib.pyplot as plt
import numpy as np

from util import load_gzip


def count_occurrences(my_list):
    # Creating an empty dictionary
    count = {}
    for i in my_list:
        count[i] = count.get(i, 0) + 1
    return count


def main():

    gloss_occurrences = load_gzip("D:/Thesis/unfiltered_gloss_occurrences.gzip")
    # gloss_occurrences = load_gzip("D:/Thesis/gloss_occurrences.gzip")

    print(f"The Corpus NGT contains {len(gloss_occurrences)} glosses, from which {len(set(gloss_occurrences))} are unique instances.")

    gloss_count = count_occurrences(gloss_occurrences)

    sorted_gloss_count = dict(sorted(gloss_count.items(), key=lambda item: item[1], reverse=True))

    n_top = 10
    # print(f"The {n_top} most frequent annotated glosses are:\n{list(sorted_gloss_count.keys())[:n_top]}")
    most_freq_summary = f"Top {n_top} most frequent glosses\n"
    most_freq_summary += "\n".join([f"{key}: {sorted_gloss_count[key]}" for key in list(sorted_gloss_count.keys())[:n_top]])

    idxs = np.arange(len(sorted_gloss_count))
    min_occurrences = 10

    f_idxs = np.where(np.array(list(sorted_gloss_count.values()), dtype=int) < min_occurrences)[0]
    f_gloss = np.array(list(sorted_gloss_count.keys()))[f_idxs]

    percentage_below = round(len(f_gloss) / len(gloss_count) * 100, 2)
    percentage_above = round(100 - percentage_below, 2)

    print(f"{len(f_gloss)} glosses appear less than 10 time in the CNGT. That is a {percentage_below}%")

    plt.figure(figsize=(16, 8))
    # plt.axhline(min_occurrences, c='r', ls='--')
    plt.axvline(f_idxs[0], c='r', ls='--')
    plt.text(x=len(idxs),
             y=0.01*max(sorted_gloss_count.values()),
             s=most_freq_summary,
             bbox={'facecolor': 'pink', 'alpha': 0.5, 'pad': 10},
             horizontalalignment='right')
    plt.text(x=f_idxs[0] + (len(idxs)-f_idxs[0])/4,
             y=0.3*max(sorted_gloss_count.values()),
             s=f"{percentage_below}%",
             bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
             horizontalalignment='center')
    plt.text(x=f_idxs[0]/2,
             y=0.3 * max(sorted_gloss_count.values()),
             s=f"{percentage_above}%",
             bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10},
             horizontalalignment='center')
    plt.bar(idxs, sorted_gloss_count.values())
    plt.xlabel("Glosses")
    plt.ylabel("Frequency")
    plt.title("Histogram of glosses in Corpus NGT")
    plt.yscale('log')
    plt.show()




if __name__ == "__main__":
    main()
