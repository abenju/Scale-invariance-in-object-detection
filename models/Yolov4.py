import torch
from torch import nn


from .yolov4_backbones.CSPDarknet53 import CSPDarknet53
from .Yolov4Necks import PanNeck
from .Yolov4Heads import Yolov4Head


class Yolov4(nn.Module):
    def __init__(self, pretrained, n_classes=80, inference=False, backbone=CSPDarknet53, neck=PanNeck, head=Yolov4Head):
        super(Yolov4, self).__init__()

        # backbone: to be changed. takes input and outputs 3 layer outputs
        self.backbone = backbone()
        # neck: Can be changed takes 3 layer outputs from the backbone and produces 3 layer outputs
        self.neck = neck()
        # head: takes 3 layer outputs and produces the output
        self.head = head(num_classes=n_classes, inference=inference)

    def forward(self, data):
        x1, x2, x3 = self.backbone(data)
        y1, y2, y3 = self.neck(x1, x2, x3)
        x = self.head(y1, y2, y3)

        return x
