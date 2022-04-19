import matplotlib.pyplot as plt
import numpy as np

from util import load_gzip

def main():
    anns_per_signer = load_gzip("D:/Thesis/anns_per_signer.gzip")
    glosses_per_signer = load_gzip("D:/Thesis/glosses_per_signer.gzip")

    anns_count = [len(values) for values in anns_per_signer.values()]
    glosses_count = [len(values) for values in glosses_per_signer.values()]

    x = np.arange(len(glosses_count))

    plt.figure(figsize=(16, 8))
    plt.bar(x+0.2, glosses_count, width=0.4, label="Glosses in Signbank")
    plt.bar(x-0.2, anns_count, width=0.4, label="Glosses")
    plt.xticks(ticks=x, labels=anns_per_signer.keys(), rotation=90)
    plt.title("Number of annotations per signer in the Corpus NGT")
    plt.ylabel("Number of annotations")
    plt.xlabel("Signer ID")
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()
