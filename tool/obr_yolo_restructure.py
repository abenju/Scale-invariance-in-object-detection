import json
import os
import random
import math
from tqdm import tqdm


"""Hyper parameters"""
master_data_dir = "/mnt/mars-alpha/aduen/cones/images"
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

output_train_path = '/mnt/mars-alpha/aduen/cones/obr_train.txt'
output_test_path = '/mnt/mars-alpha/aduen/cones/obr_test.txt'
output_val_path = '/mnt/mars-alpha/aduen/cones/obr_val.txt'

split_ratios = {  # ratios to split entire dataset. Values must equal sum up to 1
    output_train_path: 0.65,
    output_test_path: 0.2,
    output_val_path: 0.15,
}

random.seed(4)


"""Get all the annotation data from subdirectories"""
subdirs = [f.path for f in os.scandir(master_data_dir) if f.is_dir()]
frames = []
id_counter = 0

for dir in subdirs:  # for each subgroup of images
    annotation_dir = os.path.join(dir, annotation_dirs_name)
    files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
    for file in tqdm(files):  # for each annotation file (image)
        id_counter += 1
        """Load json data"""
        with open(os.path.join(annotation_dir, file), encoding='utf-8') as f:
            data = json.load(f)

        image_name = file.replace('.json', '')
        image_dir = os.path.join(dir, image_dirs_name)
        image_path = os.path.join(image_dir, image_name)  #reverse engineer the image path from the annotation path
        label_dir = image_dir.replace('images', 'labels')
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        annotations = []

        img_w = data['size']['width']
        img_h = data['size']['height']
        for object in data['objects']:  # for each annotation
            if object['classTitle'] in accepted_labels.keys():  # filters object with labels we don't want
                label = accepted_labels[object['classTitle']]  # matches string label to predefined class id
                points = object["points"]["exterior"]  # These are vertices of the bounding boxes [[xmin, ymin], [xmax, ymax]]
                x1 = points[0][0]
                x2 = points[1][0]
                y1 = points[0][1]
                y2 = points[1][1]

                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                x = (x1 / img_w) + (w / 2)
                y = (y1 / img_h) + (h / 2)

                annotations.append([label, x, y, w, h])
        if len(annotations) > 0:
            frames.append({
                'file_path': image_path,
                'label_dir': label_dir,
                'label_path': label_path,
                'annotations': annotations,
            })


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
        for line in tqdm(split[1]):
            f.write(line['file_path'])
            f.write('\n')