from torchvision.models import resnet
import torch


def resnet18(cfg):
    model = resnet.resnet18()
    return model


def resnet34(cfg):
    model = resnet.resnet34()
    return model


def resnet50(cfg):
    model = resnet.resnet50()
    return model


def resnet101(cfg):
    model = resnet.resnet101()
    return model


def resnet152(cfg):
    model = resnet.resnet152()
    return model
