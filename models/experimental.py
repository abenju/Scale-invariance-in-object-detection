# This file contains experimental modules

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv, DWConv
from utils.google_utils import attempt_download


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

class ResnetBottleneck(nn.Module):
    # Deeper Bottleneck as used in resnet 50/101/152
    def __init__(self, c1, c2, s=1, g=1, e=4, act=nn.ReLU(inplace=True)):  # ch_in, ch_out, stride, groups, expansion, activation
        super(ResnetBottleneck, self).__init__()
        c_ = int(c2 // e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c_, 3, s, g=g, act=act)
        self.conv = nn.Conv2d(c_, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = act
        self.down = None
        if s != 1 or c1 != c2:
            self.down = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(c2)
            )


    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)
        out = self.conv(out)
        out = self.bn(out)
        identity = self.down(x) if self.down is not None else x
        out += identity
        return self.act(out)

class ResNetBlock(nn.Module):
    def __init__(self, c1, c2, n=1, downsample=True, shortcut=True, g=1, e=4):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ResNetBlock, self).__init__()
        c_ = int(c2 // e)  # hidden channels
        s = 2 if downsample else 1
        if n > 1:
            self.m = nn.Sequential(*[ResnetBottleneck(c1, c2, s)]+[ResnetBottleneck(c2, c2) for _ in range(n-1)])
        else:
            self.m = ResnetBottleneck(c1, c2, s)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.m(x)


class DenseNetBottleneck(nn.Module):
    def __init__(self, c1, c2):
        super(DenseNetBottleneck, self).__init__()
        c_ = 4 * c2
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv1 = nn.Conv2d(c1, c_, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_)
        self.conv2 = nn.Conv2d(c_, c2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(nn.functional.relu(self.bn1(x)))
        out = self.conv2(nn.functional.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class DenseNetTrasition(nn.Module):
    def __init__(self, c1, c2):
        super(DenseNetTrasition, self).__init__()
        
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = nn.functional.avg_pool2d(out, 2)
        return out

class DenseBlock(nn.Module):
    def __init__(self, c1, c2, n):
        super(DenseBlock, self).__init__()
        k = (c2 - c1) // n  # growth factor
        c_ = c1
        layers = []
        for i in range(n):
            layers.append(DenseNetBottleneck(c_, k))
            c_ += k

        self.m = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(c2)
    
    def forward(self, x):
        out = self.m(x)
        out = self.bn(out)
        return nn.functional.relu(out)
