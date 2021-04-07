import json
from collections import defaultdict
from tqdm import tqdm
import os
import math
import random

random.seed(4)

ignore_unannotated = True
sample_step = 8

json_file_path = '/mnt/mars-alpha/aduen/roads/annots_12fps_full_v1.0.json'
images_dir_path = '/mnt/mars-alpha/aduen/roads/images'
output_train_path = '/mnt/mars-alpha/aduen/roads/roads_train.txt'
output_test_path = '/mnt/mars-alpha/aduen/roads/roads_test.txt'
output_val_path = '/mnt/mars-alpha/aduen/roads/roads_val.txt'

split_ratios = {  # ratios to split entire dataset. Values must equal sum up to 1
    output_train_path: 0.65,
    output_test_path: 0.2,
    output_val_path: 0.15,
}

"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

database = data['db']

frames = []

"""Etract all annotations into a list"""
for videoname in tqdm(database.keys()):
    for frame_id in database[videoname]['frames']:

        frame = database[videoname]['frames'][frame_id]
        if frame['annotated'] > 0 and (not ignore_unannotated or len(frame['annos']) > 0):  # filter frames
            file_name = "{:08}.jpg".format(int(frame_id))  # on release, image names have 5 digits
            file_dir = os.path.join(images_dir_path, videoname)
            file_path = os.path.join(file_dir, file_name)
            label_dir = file_dir.replace('images', 'labels')
            label_path = os.path.join(label_dir, file_name.replace('.jpg', '.txt'))

            width, height = frame['width'], frame['height']
            frame_annos = frame['annos']
            annotations = []
            for key in frame_annos:
                anno = frame_annos[key]
                box = anno['box']
                w = box[2] - box[0]
                h = box[3] - box[1]
                x = w / 2 + box[0]
                y = h / 2 + box[1]

                label = anno['agent_ids'][0]
                label = label - 1 if (label > 4) else label
                box_info = [label, x, y, w, h]
                annotations.append(box_info)
            frames.append({
                'file_path': file_path,
                'label_dir': label_dir,
                'label_path': label_path,
                'annotations': annotations,
            })  # All frames in same list to shuffle (only relevant annotation)


"""Shuffle and split data"""
random.shuffle(frames)
prev_index = 0
splits = []
for path, ratio in split_ratios.items():
   next_index = prev_index + math.ceil(len(frames)*ratio)
   splits.append((path, frames[prev_index:next_index]))
   prev_index = next_index

"""write lables"""
for frame in tqdm(frames):
    if not os.path.exists(frame['label_dir']):
        os.makedirs(frame['label_dir'])
    with open(frame['label_path'], 'w') as f:
        for line in frame['annotations']:
            line_str = "{} {} {} {} {}\n".format(line[0], line[1], line[2], line[3], line[4])
            f.write(line_str)
            #print(line_str)

"""write collection files"""
for split in splits:  # Write multiple files
    with open(split[0], 'w') as f:
        for i, line in tqdm(enumerate(split[1])):
            if i % sample_step == 0:
                f.write(line['file_path'])
                f.write('\n')