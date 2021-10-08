# Architecture for the self supervised model
import torch
import torch.nn.functional as F
from torch import nn
from .layers_utils import *


class FCNMultitask(nn.Module):
    def __init__(self, c_in, c_out, horizon, layers=[128, 256, 128], kss=[7, 5, 3]):
        super(FCNMultitask, self).__init__()
        assert len(layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, layers[0], kss[0])
        self.convblock2 = ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = ConvBlock(layers[1], layers[2], kss[2])
        self.gap = GAP1d(1)
        self.fc_classification = nn.Linear(layers[-1], c_out)
        self.fc_forecast = nn.Linear(layers[-1], horizon*c_in)
        self.c_in = c_in
        self.horizon = horizon

    def forward_train(self, x_cl, x_fc):
        x_cl = self.convblock1(x_cl)
        x_fc = self.convblock1(x_fc)

        x_cl = self.convblock2(x_cl)
        x_fc = self.convblock2(x_fc)

        x_cl = self.convblock3(x_cl)
        x_fc = self.convblock3(x_fc)

        x_cl = self.gap(x_cl)
        x_fc = self.gap(x_fc)

        out_classification = self.fc_classification(x_cl)
        out_forecast = self.fc_forecast(x_fc).reshape(-1, self.c_in, self.horizon)

        return out_classification, out_forecast

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        return self.fc_classification(x)
