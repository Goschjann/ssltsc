from fastcore.transform import Transform
from torch import Tensor
from scipy.interpolate import CubicSpline

import numpy as np
import torch
from torchvision import transforms
import PIL
import random
import pdb


#### Different Augmentation Strategies as Transforms
# compatible with pytorch data loaders

## helpers for the data augmentations

def random_cum_curve_generator(o, magnitude=.1, order=4, noise=None):
    x = random_curve_generator(o, magnitude=magnitude, order=order, noise=noise).cumsum()
    x -= x[0]
    x /= x[-1]
    x = np.clip(x, 0, 1)
    return x * (o.shape[-1] - 1)

def random_curve_generator(o, magnitude=.1, order=4, noise=None):
    seq_len = o.shape[-1]
    f = CubicSpline(np.linspace(-seq_len, 2 * seq_len - 1, 3 * (order - 1) + 1, dtype=int),
                    np.random.normal(loc=1.0, scale=magnitude, size=3 * (order - 1) + 1), axis=-1)
    return f(np.arange(seq_len))

def random_half_normal():
    "Returns a number between 0 and 1 with a half-normal distribution"
    while True:
        o = abs(np.random.normal(loc=0., scale=1/3))
        if o <= 1: break
    return o

def random_cum_noise_generator(o, magnitude=.1, noise=None):
    seq_len = o.shape[-1]
    x = np.clip(np.ones(seq_len) + np.random.normal(loc=0, scale=magnitude, size=seq_len), 0, 1000).cumsum()
    x -= x[0]
    x /= x[-1]
    return x * (o.shape[-1] - 1)


## functions itself


class TSMagWarp(Transform):
    "Applies warping to the y-axis of a `TSTensor` batch based on a smooth random curve"
    order = 90

    def __init__(self, magnitude=0.02, ord=4, ex=None, **kwargs):
        self.magnitude = magnitude
        self.ord = ord
        self.ex = ex
        super().__init__(**kwargs)

    def encodes(self, o):
        if self.magnitude and self.magnitude <= 0:
            return o
        y_mult = random_curve_generator(o=o, magnitude=self.magnitude, order=self.ord)
        output = o * o.new(y_mult)
        if self.ex is not None:
            output[...,self.ex,:] = o[...,self.ex,:]
        return output


class TSMagScale(Transform):
    order = 90

    def __init__(self, magnitude=0.5, ex=None, **kwargs):
        self.magnitude = magnitude
        self.ex = ex
        super().__init__(**kwargs)

    def encodes(self, o):
        if not self.magnitude or self.magnitude <= 0:
            return o
        rand = random_half_normal()
        scale = (1 - (rand  * self.magnitude)/2) if np.random.random() > 1/3 else (1 + (rand  * self.magnitude))
        output = o * scale
        if self.ex is not None:
            output[...,self.ex,:] = o[...,self.ex,:]
        return output


class TSCutOut(Transform):
    "Sets a random section of the sequence to zero"
    order = 90

    def __init__(self, magnitude=.1, ex=None, **kwargs):
        self.magnitude = magnitude
        self.ex = ex

    def encodes(self, o):
        if self.magnitude <= 0:
            return o
        seq_len = o.shape[-1]
        lambd = np.random.beta(self.magnitude, self.magnitude)
        lambd = min(lambd, 1 - lambd)
        win_len = int(seq_len * lambd)
        start = np.random.randint(-win_len + 1, seq_len)
        end = start + win_len
        start = max(0, start)
        end = min(end, seq_len)
        output = o.clone()
        output[..., start:end] = 0
        if self.ex is not None:
            output[...,self.ex,:] = o[...,self.ex,:]
        return output

class TSTimeNoise(Transform):
    order = 90

    def __init__(self, magnitude=.1, ex=None, **kwargs):
        self.magnitude = magnitude
        self.ex = ex

    def encodes(self, o):
        if self.magnitude <= 0:
            return o
        f = CubicSpline(np.arange(o.shape[-1]), o.cpu(), axis=-1)
        output = o.new(f(random_cum_noise_generator(o, magnitude=self.magnitude)))
        if self.ex is not None: output[...,self.ex,:] = o[...,self.ex,:]
        return output


class TSMagNoise(Transform):
    order = 90

    def __init__(self, magnitude=.02, ex=None, **kwargs):
        self.magnitude = magnitude
        self.ex = ex

    def encodes(self, o):
        if self.magnitude <= 0:
            return o
        seq_len = o.shape[-1]
        noise = torch.normal(0, self.magnitude, (1, seq_len), dtype=o.dtype, device=o.device)
        output = o + noise
        if self.ex is not None:
            output[...,self.ex,:] = o[...,self.ex,:]
        return output


class TSTimeWarp(Transform):
    order = 90

    def __init__(self, magnitude=.02, ord=4, ex=None, **kwargs):
        self.magnitude = magnitude
        self.ord = ord
        self.ex = ex

    def encodes(self, o):
        if self.magnitude <= 0:
            return o
        seq_len = o.shape[-1]
        f = CubicSpline(np.arange(seq_len), o.cpu(), axis=-1)
        output = o.new(f(random_cum_curve_generator(o, magnitude=self.magnitude, order=self.ord)))
        if self.ex is not None:
            output[...,self.ex,:] = o[...,self.ex,:]
        return output

class TSRandomCrop(Transform):
    "Crops a section of the sequence of a random length"
    order = 90

    def __init__(self, magnitude=.05, ex=None, **kwargs):
        self.magnitude = magnitude
        self.ex = ex


    def encodes(self, o):
        if self.magnitude <= 0:
            return o
        seq_len = o.shape[-1]
        lambd = np.random.beta(self.magnitude, self.magnitude)
        lambd = max(lambd, 1 - lambd)
        win_len = int(seq_len * lambd)
        if win_len == seq_len:
            return o
        start = np.random.randint(0, seq_len - win_len)
        output = torch.zeros(o.size(), dtype=o.dtype, device=o.device)
        output[..., start : start + win_len] = o[..., start : start + win_len]
        if self.ex is not None:
            output[...,self.ex,:] = o[...,self.ex,:]
        return output

class RandAugment:
    def __init__(self, transformations: list, num_transforms: int = 1,
                 magnitude: int = None, **kwargs):
        """
        Implementation of the RandAugment algorithm from Cubuk et al. (2020).

        Ispired by the RandAugment from the following notebook
        https://github.com/timeseriesAI/timeseriesAI/tutorial_nbs/03_Time_Series_Transforms.ipynb

        Args:
            transformations:
                A list of transformations
            num_transforms:
                The number of transformations to apply on each datapoint.
                Denoted `N` in Cubuk et al. (2020).
            magnitude:
                The magnitude of the transformations (1-10, usually 3-5).
                Denoted `M` in Cubuk et al. (2020).
            **kwargs:
        """
        self.transforms = transformations
        is_instantiated = [isinstance(tfm, Transform) for tfm in self.transforms]
        if any(is_instantiated):
            raise TypeError('One or more transformations are instantiated')

        self.num_transforms = num_transforms

        if magnitude:
            assert magnitude >= 1, 'Magnitude should be within range [1, 10]'
            assert magnitude <= 10, 'Magnitude should be within range [1, 10]'

        self.magnitude = float(min(10, magnitude)) / 10 if magnitude is not None \
            else magnitude  # Normalize the magnitude

        self.kwargs = kwargs

    def __call__(self, x):
        """
        Args:
            x:
                A time series tensor with shape
                (batch_size, n_variables, ts_length) or (n_variables, ts_length)

        Returns:
            the x transformed with num_transforms transformations
        """
        if self.num_transforms and self.magnitude and (self.num_transforms <= 0
                                                       or self.magnitude <= 0):
            return x

        selected_transforms = np.random.choice(self.transforms, self.num_transforms, replace=False)

        if self.magnitude is None:
            ready_transforms = [transform(**self.kwargs) for transform in selected_transforms]
        else:
            ready_transforms = [transform(magnitude=self.magnitude, **self.kwargs) for transform in selected_transforms]

        for tfm in ready_transforms:
            x = tfm(x)

        return x


class DuplicateTransform:
    def __init__(self, transform, duplicates: int = 2):
        """
        This transform simply wraps other transforms and applies the
        transforms on the input data `duplicates` times.

        E.g. if `x` has shape (batch_size, channels, lengths) and `duplicates=2`
        then we this transform will create a list with length 2 that contains
        duplicates of `x` with the same shape.

        This transform is needed for e.g. MixMatch (Berthelot et al., 2019)
        and MeanTeacher (Tavainen & Valpola, 2017).

        Args:
            transform:
                A transformation or a compose of several transformations
                to apply.
            duplicates:
                The number of duplicates of x to create. This parameter
                is also known as K in the Berthelot et al. (2019)
        """
        self.transform = transform
        self.duplicates = duplicates

        if transform is None and duplicates > 1:
            raise AssertionError("It does not make sense to load "
                                 "duplicates of x when no "
                                 "transforms are provided")

    def __call__(self, x):
        transformed_copies_of_x = []

        for i in range(self.duplicates):
            x_transform = self.transform(x)
            transformed_copies_of_x.append(x_transform)

        return transformed_copies_of_x


## fixmatch


PARAMETER_MAX = 10

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = PIL.Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img


class TransformFixMatch(object):
    def __init__(self, weak_transform, strong_transform):
        self.weak = weak_transform
        self.strong = strong_transform

    def __call__(self, x):
        return self.weak(x), self.strong(x)