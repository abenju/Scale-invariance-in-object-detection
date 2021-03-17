import json
import os
import random
import math
from tqdm import tqdm


"""Hyper parameters"""
master_data_dir = "D:\\Documents\\datasets\\cones\\all"
image_dirs_name = "img"
annotation_dirs_name = "ann"
accepted_labels = {  # matches a string to a class id. helpful to deal with one class being referred by multiple names
    'blue': 0,
    'blue_yolo': 0,
    'yellow': 1,
    'yellow_yolo': 1,
    'orange': 2,
    'orange_yolo': 2,
    'big_orange': 3,
    'big orange': 3,
}

output_train_path = '../data/obr_train.txt'
output_test_path = '../data/obr_test.txt'
output_val_path = '../data/obr_val.txt'

split_ratios = {  # ratios to split entire dataset. Values must equal sum up to 1
    output_train_path: 0.6,
    output_test_path: 0.2,
    output_val_path: 0.2,
}

random.seed(4)


"""Get all the annotation data from subdirectories"""
subdirs = [f.path for f in os.scandir(master_data_dir) if f.is_dir()]
lines = []

for dir in subdirs:  # for each subgroup of images
    annotation_dir = os.path.join(dir, annotation_dirs_name)
    files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
    for file in tqdm(files):  # for each annotation file (image)
        """Load json data"""
        with open(os.path.join(annotation_dir, file), encoding='utf-8') as f:
            data = json.load(f)
        image_path = os.path.join(dir, image_dirs_name, file.replace('.json', ''))  #reverse engineer the image path from the annotation path
        line = [image_path]  # image path is always the first element

        for object in data['objects']:  # for each annotation
            if object['classTitle'] in accepted_labels.keys():  # filters object with labels we don't want
                label = accepted_labels[object['classTitle']]  # matches string label to predefined class id
                points = object["points"]["exterior"]  # These are vertices of the bounding boxes [[xmin, ymin], [xmax, ymax]]
                x1 = points[0][0]
                x2 = points[1][0]
                y1 = points[0][1]
                y2 = points[1][1]

                line.append(" %d,%d,%d,%d,%d" % (x1, y1, x2, y2, label))
        if len(line) > 1:
            lines.append(line)


"""Shuffle and split data"""
random.shuffle(lines)
prev_index = 0
splits = []
for path, ratio in split_ratios.items():
    next_index = prev_index + math.ceil(len(lines)*ratio)
    splits.append((path, lines[prev_index:next_index]))
    prev_index = next_index


"""write to txt"""
for split in splits:  # Write multiple files
    with open(split[0], 'w') as f:
        for line in tqdm(split[1]):
            for item in line:
                f.write(item)
            f.write('\n')
