import argparse
import os
import shutil
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm

from utils.util import load_gzip


def main(params):
    dataset_root = params.dataset_root
    output_root = params.output_root
    cngt_vocab_path = params.cngt_vocab_path

    os.makedirs(output_root, exist_ok=True)

    cngt_vocab = load_gzip(cngt_vocab_path)

    print(f"Filtering SignBank videos to\n{output_root}")

    for file in tqdm(os.listdir(dataset_root)):
        gloss = '-'.join(file.split('-')[:-1])
        gloss = gloss.replace("#", "")
        if gloss in cngt_vocab['glosses']:
            shutil.copy(os.path.join(dataset_root, file), os.path.join(output_root, file))

    print("Zipping SignBank resized videos")

    zip_basedir = Path(output_root).parent
    zip_filename = os.path.basename(output_root) + '.zip'
    all_filenames = os.listdir(output_root)

    with ZipFile(os.path.join(zip_basedir, zip_filename), 'w') as zipfile:
        for filename in tqdm(all_filenames):
            zipfile.write(os.path.join(output_root, filename), filename)

    if os.path.isfile(os.path.join(output_root, zip_filename)):
        # maybe remove in a future
        print("Zipfile saved succesfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="D:/Thesis/datasets/NGT_Signbank_resized"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="D:/Thesis/datasets/NGT_Signbank_filtered_resized"
    )

    parser.add_argument(
        "--cngt_vocab_path",
        type=str,
        default="D:/Thesis/datasets/cngt_vocab.gzip"
    )

    params, _ = parser.parse_known_args()
    main(params)