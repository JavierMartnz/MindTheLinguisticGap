from src.utils.util import load_gzip


def main():
    new_f = load_gzip("D:/Thesis/datasets/signbank_vocab.gzip")
    old_f = load_gzip("D:/Thesis/datasets/signbank_vocab_from_csv.gzip")
    cngt_f = load_gzip("D:/Thesis/datasets/cngt_vocab.gzip")

    # print(len(new_f['glosses']))
    # print(len(old_f['words']))
    print(len(cngt_f['glosses']))

    print(len(set(new_f['glosses']).intersection(set(cngt_f['glosses']))))



if __name__ == "__main__":
    main()
