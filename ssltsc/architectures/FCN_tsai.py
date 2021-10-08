# Architecture and Code recycled from:
# https://github.com/timeseriesAI/tsai/blob/master/tsai/models/FCN.py

import torch
from torch import nn
import torch.nn.functional as F

from .layers_utils import *


class FCN(nn.Module):
    def __init__(self, c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3]):
        super(FCN, self).__init__()
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0])
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], kss[2])
        self.gap = GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        return self.fc(x)
