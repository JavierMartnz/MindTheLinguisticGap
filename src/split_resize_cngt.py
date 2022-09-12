import cv2
import os
from tqdm import tqdm
import argparse
import pympi
from multiprocessing import Pool


def get_signers_dict(ann_object):
    signers = {}

    for tier in ann_object.tiers:

        if tier[-2:] == "S1" and "PARTICIPANT" in ann_object.tiers[tier][2].keys():
            signers["S1"] = ann_object.tiers[tier][2]["PARTICIPANT"]
        elif tier[-2:] == "S2" and "PARTICIPANT" in ann_object.tiers[tier][2].keys():
            signers["S2"] = ann_object.tiers[tier][2]["PARTICIPANT"]

    return signers


def split_annotations_and_resize_videos(dataset_root, ann_filename, output_root):
    eaf_file = pympi.Elan.Eaf(os.path.join(dataset_root, ann_filename))
    # get the left/right signer and participant mappings
    signers_dict = get_signers_dict(eaf_file)
    signer_eaf_obj = {}

    # check which signers have glosses
    for tier in list(eaf_file.tiers.keys())[:4]:
        # if there are annotations for the tier
        if eaf_file.get_annotation_data_for_tier(tier):
            if "S1" in tier:
                signer_eaf_obj["S1"] = pympi.Elan.Eaf()
            if "S2" in tier:
                signer_eaf_obj["S2"] = pympi.Elan.Eaf()

    if len(signer_eaf_obj) == 0:
        # print(f"File {ann_filename} has no glosses, so not processed.")
        return

    for signer in signer_eaf_obj.keys():

        # copy all parameters except tiers and annotations
        signer_eaf_obj[signer].adocument = eaf_file.adocument
        signer_eaf_obj[signer].licenses = eaf_file.licenses
        signer_eaf_obj[signer].header = eaf_file.header
        signer_eaf_obj[signer].properties = eaf_file.properties
        signer_eaf_obj[signer].linguistic_types = eaf_file.linguistic_types
        signer_eaf_obj[signer].locales = eaf_file.locales
        signer_eaf_obj[signer].languages = eaf_file.languages
        signer_eaf_obj[signer].constraints = eaf_file.constraints
        signer_eaf_obj[signer].controlled_vocabularies = eaf_file.controlled_vocabularies
        signer_eaf_obj[signer].external_refs = eaf_file.external_refs
        signer_eaf_obj[signer].lexicon_refs = eaf_file.lexicon_refs

        # the tier and its contents can be copied directly, but when a single annotation fails then nothing
        # is copied, therefore it's safer to copy them one by one
        for tier in eaf_file.tiers.keys():
            tier_anns = eaf_file.get_annotation_data_for_tier(tier)
            # we have already filtered out the files with no annotation, so we copy every tier even if empty to keep consistency
            if signer in tier:
                signer_eaf_obj[signer].add_tier(tier)  # add tier
                for ann in tier_anns:
                    # if annotations don't have duration 0
                    if ann[0] != ann[1]:
                        signer_eaf_obj[signer].add_annotation(tier, ann[0], ann[1], ann[2])

        # remove default tier that comes when creating an elan file
        signer_eaf_obj[signer].remove_tier("default")

        new_filename = signers_dict[signer] + "_" + ann_filename.split('_')[-1]
        new_filepath = os.path.join(output_root, new_filename)

        # add linked video file, so that opening an .eaf file opens its corresponding video automatically
        new_video_filename = new_filename[:-4] + '.mpg'
        new_video_path = os.path.join(output_root, new_video_filename)
        signer_eaf_obj[signer].add_linked_file(new_video_path, new_video_path, mimetype='video/mpeg')

        # write elan object to file (this function does NOT overwrite files so we do it manually)
        if os.path.exists(new_filepath):
            os.remove(new_filepath)
        pympi.Elan.to_eaf(new_filepath, signer_eaf_obj[signer], pretty=True)

        # only split the video that corresponds to the signer
        video_path = os.path.join(dataset_root, ann_filename[:-4] + ".mpg")
        vcap = cv2.VideoCapture(video_path)
        h = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)

        # when video with only one signer just scale
        if w < 704:
            cmd = f'ffmpeg -hide_banner -loglevel error -i {video_path} -y -vf "scale=256:256" -r 25 {os.path.join(output_root, new_video_filename)}'
        # S1 is always left , S2 is right
        elif signer == "S1":
            cmd = f'ffmpeg -hide_banner -loglevel error -i {video_path} -y -vf "crop=w={w / 2}:h={h}:x={0}, scale=256:256" -r 25 {os.path.join(output_root, new_video_filename)}'
        elif signer == "S2":
            cmd = f'ffmpeg -hide_banner -loglevel error -i {video_path} -y -vf "crop=w={w / 2}:h={h}:x={w / 2}, scale=256:256" -r 25 {os.path.join(output_root, new_video_filename)}'

        os.system(cmd)


def main(params):
    dataset_root = params.dataset_root
    output_root = params.output_root

    anns_in_dir = [file for file in os.listdir(dataset_root) if file.endswith('.eaf')]
    os.makedirs(output_root, exist_ok=True)

    print("Splitting videos in CNGT with two signers...")

    # multiprocessing bit based on https://github.com/tqdm/tqdm/issues/484
    pool = Pool()
    pbar = tqdm(total=len(anns_in_dir))

    def update(*a):
        pbar.update()

    for i in range(pbar.total):
        pool.apply_async(split_annotations_and_resize_videos, args=(dataset_root, anns_in_dir[i], output_root),
                         callback=update)

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="D:/Thesis/datasets/CNGT"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="D:/Thesis/datasets/CNGT_isolated_signers"
    )

    params, _ = parser.parse_known_args()
    main(params)
