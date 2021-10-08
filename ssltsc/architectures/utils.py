"""Utils for the architectures/ backbones
"""
from functools import partial
from math import floor

import pdb
import torch
import numpy as np
import torch.nn as nn

from ssltsc.architectures import convnet13, wideresnet28, ResNet, InceptionTime, ResCNN
from ssltsc.architectures.FCN_tsai import FCN
from ssltsc.architectures.fcn_multitask import FCNMultitask
from ssltsc.architectures.convlarge import ConvLarge, ConvLargeDecoder
from ssltsc.architectures.ladder import Ladder
from ssltsc.architectures.fcn import LadderFCN, LadderFCNDecoder
from ssltsc.architectures.wideresnet28 import WideResNet28
from ssltsc.architectures.convnet13 import ConvNet13
from ssltsc.architectures.ResNet import ResNet
from ssltsc.architectures.InceptionTime import InceptionTime
from ssltsc.architectures.ResCNN import ResCNN
from ssltsc.architectures.resnetsuresort import ResnetSuresort

def backbone_factory(architecture, dataset, n_classes, n_channels, lengthts, horizon=None):
    """Creates backbone and backbone dictionary for
    instantiation of a backbone
    """
    if architecture == 'wideresnet28':
        backbone_dict = {'n_classes': n_classes}
        backbone = WideResNet28
    elif architecture == 'convnet13':
        backbone_dict = {'n_classes': n_classes}
        backbone = ConvNet13
    elif architecture == 'resnetsuresort':
        backbone_dict = {'n_classes': n_classes}
        backbone = ResnetSuresort
    elif architecture == 'FCN':
        backbone = FCN
        backbone_dict = {'c_in': n_channels,
                         'c_out': n_classes}
    elif architecture == 'resnet':
        backbone = ResNet
        backbone_dict = {'c_in': n_channels,
                         'c_out': n_classes}
    elif architecture == 'fcnmultitask':
        backbone = FCNMultitask
        backbone_dict = {'c_in': n_channels,
                         'c_out': n_classes,
                         'horizon': floor(horizon * lengthts)}
    elif architecture == 'inceptiontime':
        backbone = InceptionTime
        backbone_dict = {'c_in': n_channels,
                         'c_out': n_classes}
    elif architecture == 'rescnn':
        backbone = ResCNN
        backbone_dict = {'c_in': n_channels,
                         'c_out': n_classes}
    elif architecture == 'ladder':
        if dataset == 'cifar10':
            backbone_dict = {'n_classes': n_classes, 'channels': n_channels}
            backbone = partial(Ladder,
                               encoder_architecture=ConvLarge,
                               decoder_architecture=ConvLargeDecoder)
        else:
            backbone_dict = {'n_classes': n_classes, 'channels': n_channels}
            backbone = partial(Ladder,
                               encoder_architecture=LadderFCN,
                               decoder_architecture=LadderFCNDecoder,
                               length=lengthts
                               )

    elif architecture == 'ConvLarge':
        assert dataset == 'cifar10', 'Ladder architecture is only ' \
                                     'implemented for image data'
        backbone_dict = {'n_classes': n_classes, 'channels': n_channels}
        backbone = ConvLarge
    else:
        backbone_dict = {'n_classes': n_classes,
                         'n_variables': n_channels,
                         'length_ts': lengthts,
                         'dropout_ratio': 0.5}
        backbone = FCN

    return backbone, backbone_dict