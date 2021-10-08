"""Implementation of supervised baseline model
"""
import torch.autograd
import pandas as pd
import torch
import pdb
import numpy as np

from torch import nn, optim
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from tslearn.barycenters import softdtw_barycenter
from .losses import mixup_cross_entropy
from .basemodel import BaseModel


class Supervised(BaseModel):
    """Train backbone architecture supervised only as supervised baseline
    for ssl experiments
    """
    def __init__(self, backbone, backbone_dict, metrics=None):
        super().__init__(backbone=backbone, backbone_dict=backbone_dict)
        if metrics is not None:
            self.history.update([(metric, []) for metric in metrics])

    def __indices_to_one_hot(self, data, n_classes):
        """Convert an iterable of indices to one-hot encoded labels
        Args:
            data: {np.array} output array with the respective class
            n_classes: {int} number of classes to one-hot encoded
        Returns:
            {np.array} a one-hot encoded array
        """
        targets = data.reshape(-1)
        return torch.eye(n_classes)[targets]

    def __mixup(self, x1, x2, y1, y2, alpha=0.75, dtw=False, shuffle=False):
        """Mixup of two data points
        yields an interpolated mixture of both input samples
        """
        # shuffle (x2, y2) if x1=x2
        if torch.all(y1==y2).item():
            rand_idx = np.random.choice(a=np.arange(len(y1)), size=len(y1), replace=False)
            x2 = x2[rand_idx, ]
            y2 = y2[rand_idx]
        beta = np.random.beta(alpha, alpha)
        beta = max([beta, 1 - beta])
        if dtw:
            x = torch.empty(x1.shape)
            w1 = max([beta, 1 - beta])
            w = [w1, 1 - w1]
            for i in range(x.shape[0]):
                x[i, 0, :] = torch.tensor(softdtw_barycenter(X=[x1[i, 0, :].cpu(), x2[i, 0, :].cpu()], weights=w)[:, 0])
            y = beta * y1 + (1 - beta) * y2
            return x.to(torch.device('cuda')), y
        else:
            x = beta * x1 + (1 - beta) * x2
            y = beta * y1 + (1 - beta) * y2
            return x, y

    def train(self,
              n_steps,
              train_gen,
              opt_dict,
              val_gen=None,
              verbose=True,
              val_steps=100,
              optimizer=optim.Adam,
              es_patience=0,
              lr_scheduler=False,
              mixup=True):
        """train the model for n_steps
        """
        # objective functions for both losses
        # different loss if mixup training:
        objective_sup = nn.CrossEntropyLoss(reduction='mean')
        objective_val = nn.CrossEntropyLoss(reduction='mean')

        optimizer = optimizer(self.network.parameters(), **opt_dict)

        if torch.cuda.is_available():
            self.network.to(torch.device('cuda'))

        if lr_scheduler=='cosine':
            scheduler = CosineAnnealingLR(optimizer=optimizer,
                                          eta_min=0.0,
                                          T_max=n_steps * 1.2)
        else:
            scheduler = None

        for step in range(n_steps):
            self.network.train()

            try:
                X, Y = next(train_gen_iter)
            except:
                train_gen_iter = iter(train_gen)
                X, Y = next(train_gen_iter)
            if mixup:
            # labels to 1hot vectors
                Y = self.__indices_to_one_hot(data=Y, n_classes=train_gen.dataset.nclasses)
                X, Y = self.__mixup(x1=X, x2=X, y1=Y, y2=Y, shuffle=True)
            optimizer.zero_grad()
            if torch.cuda.is_available():
                X = X.to(torch.device('cuda'))
                Y = Y.to(torch.device('cuda'))

            Yhat = self.network(X)

            if mixup:
                loss = mixup_cross_entropy(Yhat, Y)
            else:
                loss = objective_sup(Yhat, Y)

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']
            if step % val_steps == 0 and step > 0:
                self.validate(step=step, lr=lr, train_gen=train_gen, val_gen=val_gen)
