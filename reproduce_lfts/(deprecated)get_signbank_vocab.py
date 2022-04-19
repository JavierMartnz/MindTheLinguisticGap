import os
import pandas as pd
import argparse
from src.utils.util import save_gzip

def save_vocab(word_to_id_dict, output_path, output_filename):
    save_data = {}
    save_data['words'] = list(word_to_id_dict.keys())
    save_data['word_to_id'] = word_to_id_dict
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_gzip(save_data, os.path.join(output_path, output_filename))
    print(f'{os.path.join(output_path, output_filename)} saved')
    
def get_id_from_word(dictionary, word):
    assert type(word) == str, "Input needs to be a string"
    return dictionary[word]
    
def get_word_from_id(dictionary, ann_id):
    assert type(ann_id) == int, "Input needs to be an integer"
    return list(dictionary.keys())[list(dictionary.values()).index(ann_id)]  


def main(params):
    csv_path = params.csv_path
    csv_filename = params.csv_filename
    output_path = params.output_path
    output_filename = params.output_filename
    
    signbank_df = pd.read_csv(os.path.join(csv_path, csv_filename))
    # we make sure the annotations are ordered by ID
    filtered_signbank_df = signbank_df[['Annotation ID Gloss (Dutch)', 'Signbank ID']]
    filtered_signbank_df = filtered_signbank_df.set_index('Signbank ID')
    filtered_signbank_df = filtered_signbank_df.sort_index()
    
    word_to_id_dict = {}
    for idx, word in filtered_signbank_df.itertuples():
        word_to_id_dict[word] = idx
        
    save_vocab(word_to_id_dict, output_path, output_filename)
    
if __name__ == "__main__":

    # load_data()

    # Assumes they are in the same order
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="D:/Thesis/datasets",
    )
    
    parser.add_argument(
        "--csv_filename",
        type=str,
        default="dictionary-export.csv",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="D:/Thesis/datasets",
    )
    
    parser.add_argument(
        "--output_filename",
        type=str,
        default="signbank_vocab.gzip",
    )

    params, _ = parser.parse_known_args()

    main(params)