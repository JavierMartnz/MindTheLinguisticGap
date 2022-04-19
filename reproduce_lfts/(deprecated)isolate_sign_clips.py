import os
import cv2
from src.utils.util import load_gzip
from src.utils.parse_cngt_glosses import parse_cngt_gloss_lfts
import pympi 
from intervaltree import Interval, IntervalTree
import argparse

def main(params):
        
    dataset_path = params.dataset_path
    vocab_path = params.vocab_path
    # output_path = params.output_path
    
    
    for file in os.listdir(dataset_path):
        if file.endswith('.mpg'):
            
            file_path = os.path.join(dataset_path, file)
            ann_path = file_path[:-3] + 'eaf'
            
            # check if the asociated annotation file exists
            if not os.path.exists(ann_path):
                print(f"Early return: video {file} does not have an associated annotation file")
                continue
            
            vcap = cv2.VideoCapture(file_path)
            num_video_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            if num_video_frames <= 0:
                print('Early return: video has zero frames')
                continue
            
            vocab = load_gzip(vocab_path)
            
            ann_file = pympi.Elan.Eaf(ann_path)
            
            glosses_lefth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[0])
            glosses_righth = ann_file.get_annotation_data_for_tier(list(ann_file.tiers.keys())[1])
            
            glosses_id_frame = [-1] * num_video_frames
            glosses_frame = [""] * num_video_frames
            
            left_intervalTree = IntervalTree()
            right_intervalTree = IntervalTree()
            
            if glosses_lefth:
 
                for ann in glosses_lefth:
                    start_ms, end_ms = ann[0], ann[1]
                    gloss = ann[2]
                    parsed_gloss = parse_cngt_gloss_lfts(gloss, vocab['words'])
                    # if parsed_gloss is not None, add to intervalTree
                    if parsed_gloss:
                        # calculate start and end frame with fps
                        interval = Interval(begin=start_ms, end=end_ms, data=parsed_gloss)  
                        left_intervalTree.add(interval)
                        
            if glosses_righth:                
                
                for ann in glosses_righth:
                    start_ms, end_ms = ann[0], ann[1]
                    gloss = ann[2]
                    parsed_gloss = parse_cngt_gloss_lfts(gloss, vocab['words'])
                    # if parsed_gloss is not None
                    if parsed_gloss:
                        interval = Interval(begin=start_ms, end=end_ms, data=parsed_gloss)
                        right_intervalTree.add(interval)
                        
                        
                    # these code piece stops duplicated annotation of glosses in different hands
                    # overlaps = left_intervalTree.overlap(start_ms, end_ms)
                    # if overlaps:
                    #     overlap_exceeded = False
                    #     for interval in overlaps:
                    #         intersection = min(end_ms, interval.end) - max(start_ms, interval.begin)
                    #         union = max(end_ms, interval.end) - min(start_ms, interval.begin)
                    #         iou = intersection / union
                    #         # if the gloss overlaps a lot with one in the intervalTree, skip gloss
                    #         if iou >= 0.9:
                    #             overlap_exceeded = True
                    #             break
                    #     if overlap_exceeded:
                    #         continue
                
            # we build a final tree where union of intervals is taken (to account for whole sign)
            final_intervalTree = IntervalTree()
                    
            for interval_right in right_intervalTree:
                already_in_left = left_intervalTree.overlap(interval_right.begin, interval_right.end)
                # for an interval that overlaps for the same sign
                for interval_left in already_in_left:         
                    if interval_left.data == interval_right.data:
                        new_begin = min(interval_left.begin, interval_right.begin)
                        new_end = max(interval_left.end, interval_right.end)
                        interval = Interval(begin=new_begin, end=new_end, data=interval_left.data)
                        final_intervalTree.add(interval)
            
            # HERE THE ORIGINAL SOLUTION STORES AN ARRAY WITHT THE FRAMES WHERE A SIGN OCURRS
            # THIS ALLOWS NO OVERLAP WHATSOEVER, SO IT WOULD MAKE MORE SENSE TO STORE THE 
            # INTERVALTREE AND WHENEVER THE SPLITTING IS DONE, THEN TAKE THE TIMESTAMPS THERE
            # DIRECTLY FROM THE TREE, WITHOUT REMOVE THE OVERLAPS
            
            #only one file    
            if file.endswith('.eaf'):
                break
            
                
            
        
                        
    # dataset_path = params.dataset_path
    # dest_path = params.dest_path
    
    # files_in_dir = os.listdir(dataset_path)
    # os.makedirs(dest_path, exist_ok=True)

    # # multiprocessing bit based on https://github.com/tqdm/tqdm/issues/484
    # pool = Pool()
    # pbar = tqdm(total=len(files_in_dir))
    
    # def update(*a):
    #     pbar.update()
    
    # for i in range(pbar.total):
    #     pool.apply_async(split_file, args=(files_in_dir[i], dataset_path, dest_path), callback=update)
        
    # pool.close()
    # pool.join()
        

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="D:/Thesis/datasets/NGT_split/"
    )
    
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="D:/Thesis/datasets/signbank_vocab.gzip"
    )
    
    params, _ = parser.parse_known_args()
    main(params)