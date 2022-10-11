import os
import argparse
from utils.util import save_gzip
import pandas as pd

def main(params):
    signbank_root = params.signbank_root
    output_path = params.output_path
    signbank_csv = params.signbank_csv

    sb_df = pd.read_csv("D:/Thesis/dictionary-export.csv")

    gloss_to_id = pd.Series(sb_df["Signbank ID"].values, index=sb_df["Annotation ID Gloss (Dutch)"]).to_dict()
    id_to_gloss = pd.Series(sb_df["Annotation ID Gloss (Dutch)"].values, index=sb_df["Signbank ID"]).to_dict()

    sb_vocab = {"gloss_to_id": gloss_to_id, "id_to_gloss": id_to_gloss}

    save_gzip(sb_vocab, output_path)
    print(f'Vocab saved at {output_path}')

    # all_filenames = []
    # for x in os.walk(signbank_root):
    #     all_filenames.extend(x[-1])
    #
    # gloss_to_id_dict = {}
    # for filename in all_filenames:
    #     # make sure the only files read are the sign videos
    #     if filename.endswith(".mp4"):
    #         id = int(".".join(filename.split('.')[:-1]).split('-')[-1])
    #         gloss = "-".join(filename.split('-')[:-1])
    #         gloss_to_id_dict[gloss] = id


    
if __name__ == "__main__":

    # load_data()

    # Assumes they are in the same order
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signbank_root",
        type=str,
        default="D:/Thesis/datasets/NGT_Signbank",
    )

    parser.add_argument(
        "--signbank_csv",
        type=str,
        default="D:/Thesis/dictionary_export.csv",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="D:/Thesis/datasets/signbank_vocab.gzip",
    )

    params, _ = parser.parse_known_args()

    main(params)