import os
import argparse
from utils.util import save_gzip
import pandas as pd


def main(params):
    output_path = params.output_path
    signbank_csv = params.signbank_csv

    sb_df = pd.read_csv(signbank_csv)

    english_to_dutch = pd.Series(sb_df["Annotation ID Gloss (Dutch)"].values, index=sb_df["Annotation ID Gloss (English)"]).to_dict()
    gloss_to_id = pd.Series(sb_df["Signbank ID"].values, index=sb_df["Annotation ID Gloss (Dutch)"]).to_dict()
    id_to_gloss = pd.Series(sb_df["Annotation ID Gloss (Dutch)"].values, index=sb_df["Signbank ID"]).to_dict()

    sb_vocab = {"english_to_dutch": english_to_dutch,
                "gloss_to_id": gloss_to_id,
                "id_to_gloss": id_to_gloss}

    save_gzip(sb_vocab, output_path)
    print(f'Vocab saved at {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--signbank_csv",
        type=str,
        default="D:/Thesis/dictionary-export.csv",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="D:/Thesis/datasets/signbank_vocab.gzip",
    )

    params, _ = parser.parse_known_args()

    main(params)
