import json
from collections import defaultdict
from tqdm import tqdm
import os
import math

"""hyper parameters"""
json_file_path = 'D:/Downloads/annots_12fps_full_v1.01111.json'
images_dir_path = 'D:/Downloads/rgb-images'
output_path = '../data/roads.txt'
subsets = ['all']


def is_part_of_subsets(split_ids, SUBSETS):
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True

    return is_it


"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

database = data['db']

lines = []

for videoname in tqdm(sorted(database.keys())):
    if not is_part_of_subsets(data['db'][videoname]['split_ids'], subsets):
        continue

    #numf = database[videoname]['numf']
    frames = database[videoname]['frames']
    frame_nums = [int(f) for f in frames.keys()]

    for frame_num in sorted(frame_nums):
        line = []
        frame_id = str(frame_num)
        frame_file_id = "{:08}".format(frame_num)
        if frame_id in frames.keys() and frames[frame_id]['annotated'] > 0:

            path = os.path.join(images_dir_path, videoname, frame_file_id)
            line.append(path)

            frame = frames[frame_id]
            width, height = frame['width'], frame['height']
            frame_annos = frame['annos']
            for key in frame_annos:
                anno = frame_annos[key]
                box = anno['box']
                x1 = math.floor(box[0] * width)
                y1 = math.floor(box[1] * height)
                x2 = math.floor(box[2] * width)
                y2 = math.floor(box[3] * width)

                label = anno['agent_ids'][0]
                box_info = " %d,%d,%d,%d,%d" % (x1, y1, x2, y2, label)
                line.append(box_info)
            lines.append(line)


"""write to txt"""
with open(output_path, 'w') as f:
    for line in tqdm(lines):
        for item in line:
            f.write(item)
        f.write('\n')