import torch
from torch import nn


from .yolov4_backbones.CSPDarknet53 import CSPDarknet53
from .Yolov4Necks import PanNeck
from .Yolov4Heads import Yolov4Head


class Yolov4(nn.Module):
    def __init__(self, pretrained, n_classes=80, inference=False, backbone=CSPDarknet53, neck=PanNeck, head=Yolov4Head, freeze_backbone=False):
        super(Yolov4, self).__init__()

        # backbone: to be changed. takes input and outputs 3 layer outputs
        self.backbone = backbone()
        # neck: Can be changed takes 3 layer outputs from the backbone and produces 3 layer outputs
        self.neck = neck()
        # head: takes 3 layer outputs and produces the output
        self.head = head(num_classes=n_classes, inference=inference)

        if pretrained:  # load pretrained weights from differently formatted file
            backbone_pre = {}
            neck_pre = {}
            values = torch.load(pretrained)
            for key, value in values.items():
                if key.startswith('neek.'):
                    neck_pre[key.replace('neek.', '')] = value
                elif key.startswith('down'):
                    backbone_pre[key] = value

            self.backbone.load_state_dict(backbone_pre)
            self.neck.load_state_dict(neck_pre)

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False


    def forward(self, data):
        x1, x2, x3 = self.backbone(data)
        y1, y2, y3 = self.neck(x1, x2, x3)
        x = self.head(y1, y2, y3)

        return x
