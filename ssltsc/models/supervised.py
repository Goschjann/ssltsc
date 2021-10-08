"""Implementation of supervised baseline model
"""
import torch.autograd
import pandas as pd
import time
import torch
import pdb
import numpy as np

from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tslearn.barycenters import softdtw_barycenter
from .losses import mixup_cross_entropy
from .basemodel import BaseModel
from torch.utils.data import DataLoader

from .utils import calculate_classification_metrics

class Supervised(BaseModel):
    """Train backbone architecture supervised only as supervised baseline
    for ssl experiments
    """
    def __init__(self, backbone, backbone_dict, callbacks=None):
        super().__init__(backbone=backbone, backbone_dict=backbone_dict, callbacks=callbacks)

    def _indices_to_one_hot(self, data, n_classes):
        """Convert an iterable of indices to one-hot encoded labels
        Args:
            data: {np.array} output array with the respective class
            n_classes: {int} number of classes to one-hot encoded
        Returns:
            {np.array} a one-hot encoded array
        """
        targets = data.reshape(-1)
        return torch.eye(n_classes)[targets]

    def _mixup(self, x1, x2, y1, y2, alpha=0.75, dtw=False, shuffle=False):
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

    def _validate_one_dataset(self, data_loader: DataLoader, step: int):
        """
        Helper method that
        Args:
            data_loader:
                DataLoader to calculate metrics and losses on.
            step:
                The step in the trainin loop. This is useds for the loss calculation.

        Returns:
            The dict of metrics and the average total loss and average
            reconstructions losses over the batches.
        """
        yhat_prob, yhat_logits, y = self.predict(data_loader)
        # calculate general metrics
        metrics = calculate_classification_metrics(yhat_prob, y)

        # calculate the 'originally' reduced loss
        logloss = nn.CrossEntropyLoss(weight=None, reduction='sum')

        metrics['loss'] = logloss(torch.tensor(yhat_logits), torch.tensor(y).long()).item() / len(data_loader.dataset)

        return metrics

    def train(self,
              opt_dict,
              data_dict,
              model_params,
              exp_params,
              optimizer=optim.Adam):
        """train the model for n_steps
        """
        # objective functions for both losses
        # different loss if mixup training:
        objective_sup = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)

        optimizer = optimizer(self.network.parameters(), **opt_dict)

        if exp_params['lr_scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer=optimizer,
                                          eta_min=0.0,
                                          T_max=exp_params['n_steps'] * 1.2)
        else:
            scheduler = None

        for step in range(exp_params['n_steps']):
            self.network.train()

            for cb in self.callbacks:
                cb.on_train_batch_start()

            try:
                X, Y = next(train_gen_iter)
            except:
                train_gen_iter = iter(data_dict['train_gen_l'])
                X, Y = next(train_gen_iter)

            if model_params['mixup']:
            # labels to 1hot vectors
                Y = self._indices_to_one_hot(data=Y, n_classes=train_gen.dataset.nclasses)  # FIXME: The train_gen is not defined??
                X, Y = self._mixup(x1=X, x2=X, y1=Y, y2=Y, shuffle=True)
            optimizer.zero_grad()
            if torch.cuda.is_available():
                X = X.to(torch.device('cuda'))
                Y = Y.to(torch.device('cuda'))

            Yhat = self.network(X)
            if model_params['mixup']:
                loss = mixup_cross_entropy(Yhat, Y.long())
            else:
                loss = objective_sup(Yhat, Y.long())

            loss.backward()
            optimizer.step()
            print(f'Optimizer step {step}')
            if scheduler is not None:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']

            for cb in self.callbacks:
                cb.on_train_batch_end(step=step)

            if step % exp_params['val_steps'] == 0 and step > 0:
                best_metric = 0.0 if len(self.history) == 0 else max(self.history[exp_params['early_stopping_metric']])
                metrics = self.validate(step=step,
                                        hp_dict={'lr': lr},
                                        train_dataloader=data_dict['train_gen_val'],
                                        val_dataloader=data_dict['val_gen'])

                # early stopping
                if exp_params['early_stopping'] and metrics[exp_params['early_stopping_metric']] > best_metric:
                    self.save_checkpoint(step=step, verbose=True)

        # Training is over
        for callback in self.callbacks:
            callback.on_train_end(self.history)
