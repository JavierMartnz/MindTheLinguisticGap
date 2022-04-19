from src.utils.util import load_gzip, make_dir
import argparse
import os
import shutil


def main(params):
    dataset_root = params.dataset_root
    cngt_glosses_path = params.cngt_glosses_path
    output_root = params.output_root
    log_filename = params.log_filename

    make_dir(output_root)

    cngt_glosses_set = load_gzip(cngt_glosses_path)

    if os.path.isfile(log_filename):
        os.remove(log_filename)

    with open(log_filename, 'w') as f:

        cnt_match = 0
        cnt_unmatch = 0

        for filename in os.listdir(dataset_root):

            signbank_gloss = filename.replace(filename.split('-')[-1], "")[:-1]
            # remove the # from fingerspelled glosses
            if '#' in signbank_gloss:
                signbank_gloss = signbank_gloss[1:]

            if signbank_gloss in cngt_glosses_set:
                output_path = os.path.join(output_root, filename)
                input_file = os.path.join(dataset_root, filename)
                shutil.copy(input_file, output_path)
                cnt_match += 1
                print('{} copied'.format(signbank_gloss), file=f)
            else:
                cnt_unmatch += 1
                print('{} is not in the corpus ngt glosses'.format(signbank_gloss), file=f)

        print(f'\n\n{cnt_match} files were copied and {cnt_unmatch} ignored', file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="D:/Thesis/datasets/Signbank_resized",
    )
    parser.add_argument(
        "--cngt_glosses_path",
        type=str,
        default="D:/Thesis/datasets/cngt_glosses_set.gzip",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="D:/Thesis/datasets/Signbank_resizedfiltered"
    )

    parser.add_argument(
        "--log_filename",
        type=str,
        default="D:/Thesis/datasets/signbank_filtering_results.txt"
    )

    params, _ = parser.parse_known_args()

    main(params)
