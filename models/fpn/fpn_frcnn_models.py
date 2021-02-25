import torch
import torchvision

def resnet_50_frcnn_fpn(cfg):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    return model

