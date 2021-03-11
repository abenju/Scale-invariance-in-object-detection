import json
from collections import defaultdict
from tqdm import tqdm
import os
import math
import random

"""hyper parameters"""
ignore_unannotated = True

json_file_path = 'D:/Downloads/annots_12fps_full_v1.01111.json'
images_dir_path = 'D:/Downloads/rgb-images'
output_train_path = '../data/roads_train.txt'
output_test_path = '../data/roads_test.txt'
output_val_path = '../data/roads_val.txt'

split_ratios = {  # ratios to split entire dataset. Values must equal sum up to 1
    output_train_path: 0.7,
    output_test_path: 0.15,
    output_val_path: 0.15,
}

random.seed(4)


"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

database = data['db']


lines = []
frames = []

"""Etract all annotations into a list"""
for videoname in tqdm(database.keys()):
    for frame_id in database[videoname]['frames']:

        frame = database[videoname]['frames'][frame_id]
        if frame['annotated'] > 0 and (not ignore_unannotated or len(frame['annos']) > 0):  # filter frames
            file_name = "{:08}.jpg".format(int(frame_id))
            file_path = os.path.join(images_dir_path, videoname, file_name)

            width, height = frame['width'], frame['height']
            frame_annos = frame['annos']
            annotations = []
            for key in frame_annos:
                anno = frame_annos[key]
                box = anno['box']
                x1 = math.floor(box[0] * width)
                y1 = math.floor(box[1] * height)
                x2 = math.floor(box[2] * width)
                y2 = math.floor(box[3] * width)

                label = anno['agent_ids'][0]
                box_info = " %d,%d,%d,%d,%d" % (x1, y1, x2, y2, label)
                annotations.append(box_info)
            frames.append([file_path] + annotations)  # All frames in same list to shuffle (only relevant annotation)


"""Shuffle and split data"""
random.shuffle(frames)
prev_index = 0
splits = []
for path, ratio in split_ratios.items():
   next_index = prev_index + math.ceil(len(frames)*ratio)
   splits.append((path, frames[prev_index:next_index]))
   prev_index = next_index


"""write to txt"""
for split in splits:  # Write multiple files
    with open(split[0], 'w') as f:
        for line in tqdm(split[1]):
            for item in line:
                f.write(item)
            f.write('\n')
