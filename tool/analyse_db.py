from matplotlib import pyplot as plt
import numpy as np

annotations_file = '../data/roads_train.txt'

lines = []

with open(annotations_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = line.split(' ')
        boxes = []
        for box in data[2:]:
            box_data = box.split(',')
            boxes.append({
                'x1': int(box_data[0]),
                'y1': int(box_data[1]),
                'x2': int(box_data[2]),
                'y2': int(box_data[3]),
                'label': int(box_data[4]),
            })
        image = {
            'id': data[0],
            'path': data[1],
            'boxes': boxes,
        }
        lines.append(image)

areas = []
for image in lines:
    for box in image['boxes']:
        area = (box['x2'] - box['x1'])*(box['y2'] - box['y1'])
        areas.append(area)

x = np.array(areas)

q25, q75 = np.percentile(x,[.25,.75])
bin_width = 2*(q75 - q25)*len(x)**(-1/3)
bins = round((x.max() - x.min())/bin_width)
print("Freedman–Diaconis number of bins:", bins)
plt.hist(x, bins = bins)
plt.show()
