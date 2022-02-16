import json
import os
import pdb
import torch
import numpy as np
import pandas as pd
import tsfresh
from torch.utils import data


class ImageDataset(data.Dataset):
    base_path = None

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 task='classification',
                 val_size=None,
                 test_size=None,
                 normalize=False, standardize=False,
                 scale_overall=False, scale_channelwise=False):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.labelled_idxs = None
        self.transform = transform
        self.target_transform = target_transform


    @property
    def size(self):
        return self.x.shape[2]

    @property
    def length(self):
        return None

    @property
    def nvariables(self):
        return None

    @property
    def nclasses(self):
        return len(np.unique(self.y))

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform:
            x = torch.tensor(x).to(torch.float32)
            x = self.transform(x)

        if self.labelled_idxs and index not in self.labelled_idxs:
            y = -1

        return x, y

    def __len__(self):
        raise NotImplementedError

    def load_dataset(self, part='train'):
        path = os.path.join(self.root, self.base_folder)

        x_path = os.path.join(path, 'X_{}.npy'.format(part))
        x = np.load(file=x_path).astype('float32')

        y = np.load(file=os.path.join(path, 'Y_{}.npy'.format(part))).astype('int')

        if -1 not in np.unique(y):
            classes = np.unique(y)
            for idx in range(len(classes)):
                np.place(y, y == classes[idx], idx)
        y = y.astype(int)

        assert len(x) == len(y)

        return x, y
