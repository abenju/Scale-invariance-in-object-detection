import os
import shutil
import yaml
import json
from pathlib import Path
import cv2
import tqdm

"""
Given an expected yolov5 structure (that expected by this project) it restructures the dataset to match COCO
and creates a COCO style annotation file. This is only used once for data preparation during development (manually)

this is done to make full use of pycocotools and its evaluation as well as having a standard structure and provide
id to images. NOTE: this process changes the names of files to that of a unique numerical id.
"""

data_yml_path = '/mnt/mars-alpha/aduen/Spatio-temporal-object-detection/data/obr_cones.yaml'
output_dir = '/mnt/mars-alpha/aduen/coco/obr_cones_test'
image_dir_name = 'images'
output_json_name = 'obr_cones_test.json'

#  Get dataset info
with open(data_yml_path) as f:
    ds_data = yaml.load(f, Loader=yaml.SafeLoader)

nc = ds_data['nc']
c_names = ds_data['names']
#img_listing_files = [ds_data['train'], ds_data['val'], ds_data['test']]
img_listing_files = [ds_data['test']]

#  Get list of all images
img_path_listing = []
for file in img_listing_files:
    path = Path(file)
    with open(file, 'r') as f:
        t = f.read().strip().splitlines()
        parent = str(path.parent) + os.sep
        img_path_listing += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path

images = []
annotations = []
categories = []

for i in range(nc):
    categories.append({
        "id": i + 1,
        "name": c_names[i],
    })

new_path_listing = []

#  For each image
for idx, path in enumerate(tqdm.tqdm(img_path_listing)):
    label_path = path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
    new_name = "{:07}.jpg".format(idx + 1) if path.endswith('.jpg') else "{:07}.png".format(idx + 1)
    new_path = os.path.join(output_dir, image_dir_name, new_name)
    new_path_listing.append(new_path)

    im = cv2.imread(path)
    height, width, c = im.shape

    images.append({
        "file_name": new_name,
        "id": idx + 1,
        "height": height,
        "width": width,
    })

    #os.rename(path, new_path)
    #shutil.copyfile(path, new_path)

    with open(label_path) as f:
        t = f.read().strip().splitlines()
    
    for line in t:
        cat, x, y, w, h = [float(x) for x in line.split(' ')]
        x = int(x * width)
        y = int(y * height)
        w = int(w * width)
        h = int(h * height)
        annotations.append({
            "category_id": int(cat),
            "image_id": idx + 1,
            "bbox": [int(x-(w/2)), int(y-(h/2)), w, h ],
            "area": w * h,
            "iscrowd": 0,
        })
    
    new_label_path = new_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
    with open(new_label_path, 'w') as f:
        for line in t:
            f.write(line)
            f.write('\n')

for idx, ann in enumerate(annotations):
    annotations[idx]['id'] = idx + 1  # give annotations an id


jdict = {
    "images": images,
    "annotations": annotations,
    "categories": categories,
}

jpath = os.path.join(output_dir, output_json_name)
with open(jpath, 'w') as f:
    json.dump(jdict, f)

with open(img_listing_files[0], 'w') as f:
    for line in new_path_listing:
        f.write(line)
        f.write('\n')