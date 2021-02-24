import torch.nn as nn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
import torchvision

from models.resnet101 import resnet101


def create_resnet101_faster_rcnn(num_classes=1000,
                                  means=[0.485, 0.456, 0.406],
                                  stds=[0.229, 0.224, 0.225],
                                  anchor_sizes=((32, 64, 128),),
                                  anchor_ratios=((0.5, 1.0, 2.0),)):

    #  takes resnet 101, removes the fc layer and configures out channels for the backbone
    resnet = resnet101(pretrained=True)
    #resnet = torchvision.models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]  # -2 for feature maps, -1 for average pooling
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_ratios)
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    #model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
    #                   box_roi_pool=roi_pooler, image_mean=means, image_std=stds)

    model = FasterRCNN(backbone, num_classes=num_classes, image_mean=means, image_std=stds, rpn_anchor_generator=anchor_generator)

    return model
