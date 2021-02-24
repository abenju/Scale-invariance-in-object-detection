#%%
import torch
from models import create_resnet101_faster_rcnn

import config.faster_rcnn_res101_config as config
#%%
dummy_data = torch.rand((4,3, 256, 300))
model = create_resnet101_faster_rcnn(num_classes=config.NUM_CLASSES)

#%%
