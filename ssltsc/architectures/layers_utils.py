
import torch
from torch import nn
from functools import partial
from enum import Enum
import pdb
import re
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, ni, nf, kernel_size=None, ks=3, stride=1,
                 padding='same', bias=None, bias_std=0.01, norm='Batch',
                 zero_norm=False, bn_1st=True,
                 act=nn.ReLU, act_kwargs={}, init='auto',
                 dropout=0., xtra=None, coord=False, separable=False,
                 **kwargs):

        kernel_size = kernel_size or ks
        ndim = 1
        layers = [AddCoords1d()] if coord else []
        norm_type = getattr(NormType, f"{snake2camel(norm)}{'Zero' if zero_norm else ''}") if norm is not None else None
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None:
            bias = not (bn or inn)
        if separable:
            conv = SeparableConv1d(ni + coord, nf, ks=kernel_size, bias=bias, stride=stride, padding=padding, **kwargs)
        else:
            conv = Conv1d(ni + coord, nf, ks=kernel_size, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act is None else act(**act_kwargs)
        if not separable:
            init_linear(conv, act, init=init, bias_std=bias_std)
        if norm_type == NormType.Weight:
            conv = weight_norm(conv)
        elif norm_type == NormType.Spectral:
            conv = spectral_norm(conv)
        layers += [conv]
        act_bn = []
        if act is not None:
            act_bn.append(act)
        if bn:
            act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn:
            act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st:
            act_bn.reverse()
        if dropout:
            layers += [nn.Dropout(dropout)]
        layers += act_bn
        if xtra:
            layers.append(xtra)
        super().__init__(*layers)


class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.gap(x))


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class Add(nn.Module):
    def forward(self, x, y):
        return x.add(y)
    def __repr__(self):
        return f'{self.__class__.__name__}'


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat(*x, dim=self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'



def Norm(nf, ndim=1, norm='Batch', zero_norm=False, init=True, **kwargs):
    "Norm layer with `nf` features and `ndim` with auto init."
    assert 1 <= ndim <= 3
    nl = getattr(nn, f"{snake2camel(norm)}Norm{ndim}d")(nf, **kwargs)
    if nl.affine and init:
        nl.bias.data.fill_(1e-3)
        nl.weight.data.fill_(0. if zero_norm else 1.)
    return nl


BN1d = partial(Norm, ndim=1, norm='Batch')
ConvBN = partial(ConvBlock, norm='Batch', act=None)

NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')

def snake2camel(name):
    return re.sub(r'(?:^|_)([a-z])', lambda x: x.group(1).upper(), name)


class Pad1d(nn.ConstantPad1d):
    def __init__(self, padding, value=0.):
        super().__init__(padding, value)


class Conv1dSame(nn.Module):
    "Conv1d with padding='same'"
    def __init__(self, ni, nf, ks=3, stride=1, dilation=1, **kwargs):
        self.ks, self.stride, self.dilation = ks, stride, dilation
        self.conv1d_same = nn.Conv1d(ni, nf, ks, stride=stride, dilation=dilation, **kwargs)
        self.weight = self.conv1d_same.weight
        self.bias = self.conv1d_same.bias
        self.pad = Pad1d

    def forward(self, x):
        self.padding = same_padding1d(x.shape[-1], self.ks, dilation=self.dilation) #stride=self.stride not used in padding calculation!
        return self.conv1d_same(self.pad(self.padding)(x))


def Conv1d(ni, nf, kernel_size=None, ks=None, stride=1, padding='same', dilation=1, init='auto', bias_std=0.01, **kwargs):
    "conv1d layer with padding='same', 'causal', 'valid', or any integer (defaults to 'same')"
    assert not (kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'
    kernel_size = kernel_size or ks
    if padding == 'same':
        if kernel_size%2==1:
            conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=kernel_size//2 * dilation, dilation=dilation, **kwargs)
        else:
            conv = Conv1dSame(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'causal': conv = Conv1dCausal(ni, nf, kernel_size, stride=stride, dilation=dilation, **kwargs)
    elif padding == 'valid': conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
    else: conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=padding, dilation=dilation, **kwargs)
    init_linear(conv, None, init=init, bias_std=bias_std)
    return conv


class SeparableConv1d(nn.Module):
    def __init__(self, ni, nf, ks, stride=1, padding='same', dilation=1, bias=True, bias_std=0.01):
        self.depthwise_conv = Conv1d(ni, ni, ks, stride=stride, padding=padding, dilation=dilation, groups=ni, bias=bias)
        self.pointwise_conv = nn.Conv1d(ni, nf, 1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        if bias:
            if bias_std != 0:
                normal_(self.depthwise_conv.bias, 0, bias_std)
                normal_(self.pointwise_conv.bias, 0, bias_std)
            else:
                self.depthwise_conv.bias.data.zero_()
                self.pointwise_conv.bias.data.zero_()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


def init_linear(m, act_func=None, init='auto', bias_std=0.01):
    if getattr(m,'bias',None) is not None and bias_std is not None:
        if bias_std != 0: normal_(m.bias, 0, bias_std)
        else: m.bias.data.zero_()
    if init=='auto':
        if act_func in (F.relu_,F.leaky_relu_): init = kaiming_uniform_
        else: init = getattr(act_func.__class__, '__default_init__', None)
        if init is None: init = getattr(act_func, '__default_init__', None)
    if init is not None: init(m.weight)


def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, ndim, zero=norm_type==NormType.BatchZero, **kwargs)

def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn


def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a
