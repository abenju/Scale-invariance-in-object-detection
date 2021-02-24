import torch
import torch.nn as nn
import torchvision

from torch.nn.functional import softmax


class AutofocusNet(nn.Module):
    def __init__(self):
        super(AutofocusNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(256, 2, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        shape = x.size()
        x = torch.reshape(x, (shape[0], 2, -1))
        x = softmax(x, 2)

        return x

