"""Implementation of the self-supervised model by Jawed et al. 2020
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

class SelfSupervised(BaseModel):
    """Train backbone architecture supervised only as supervised baseline
    for ssl experiments
    """
    def __init__(self, backbone, backbone_dict, callbacks=None):
        super().__init__(backbone=backbone, backbone_dict=backbone_dict, callbacks=callbacks)

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

        metrics['loss'] = logloss(torch.tensor(yhat_logits), torch.tensor(y)).item() / len(data_loader.dataset)

        return metrics

    def train(self,
              opt_dict,
              data_dict,
              model_params,
              exp_params,
              optimizer=optim.Adam):
        """train the model for n_steps
        """

        assert data_dict['train_gen_forecast'] is not None, "You need to give me a forecasting data loader"

        # objective functions for supervised and self-supervised loss
        objective_sup = nn.CrossEntropyLoss(reduction='mean')

        objective_forecast = nn.MSELoss(reduction='mean')

        optimizer = optimizer(self.network.parameters(), **opt_dict)

        if exp_params['lr_scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer=optimizer,
                                          eta_min=0.0,
                                          T_max=exp_params['n_steps'] * 1.2)
        else:
            scheduler = None

        train_tracking_loss = train_cl_loss = train_fc_loss = 0.0
        for step in range(exp_params['n_steps']):
            self.network.train()

            for cb in self.callbacks:
                cb.on_train_batch_start()

            # get the classification data
            try:
                X, Y = next(train_gen_classification_iter)
            except:
                train_gen_classification_iter = iter(data_dict['train_gen_l'])
                X, Y = next(train_gen_classification_iter)

            # get the forecasting data
            try:
                X_fc, Y_fc = next(train_gen_forecast_iter)
            except:
                train_gen_forecast_iter = iter(data_dict['train_gen_forecast'])
                X_fc, Y_fc = next(train_gen_forecast_iter)

            optimizer.zero_grad()
            if torch.cuda.is_available():
                X = X.to(torch.device('cuda'))
                Y = Y.to(torch.device('cuda'))
                X_fc = X_fc.to(torch.device('cuda'))
                Y_fc = Y_fc.to(torch.device('cuda'))

            Yhat_cl, Yhat_fc = self.network.forward_train(X, X_fc)

            loss_cl = objective_sup(Yhat_cl, Y)
            loss_fc = objective_forecast(Yhat_fc, Y_fc)
            loss = loss_cl + model_params['lambda'] * loss_fc

            # log losses
            train_tracking_loss += loss.item()
            train_cl_loss += loss_cl.item()
            train_fc_loss += loss_fc.item()

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']

            for cb in self.callbacks:
                cb.on_train_batch_end(step=step)

            if step % exp_params['val_steps'] == 0 and step > 0:
                best_metric = 0.0 if len(self.history) == 0 else max(self.history[exp_params['early_stopping_metric']])
                print(f'Step {step}, loss {round(loss.item(), 5)}, class loss {round(loss_cl.item(), 5)}, forecast loss {round(loss_fc.item(), 5)}')
                metrics = self.validate(step=step,
                                        hp_dict={'lr': lr,
                                                 'train_tracking_loss_cl': train_cl_loss / exp_params['val_steps'],
                                                 'train_tracking_loss_fc': train_fc_loss / exp_params['val_steps'],
                                                 'train_tracking_loss': train_tracking_loss / exp_params['val_steps']},
                                        train_dataloader=data_dict['train_gen_val'],
                                        val_dataloader=data_dict['val_gen'])

                # early stopping
                if exp_params['early_stopping'] and metrics[exp_params['early_stopping_metric']] > best_metric:
                    self.save_checkpoint(step=step, verbose=True)

                train_tracking_loss = train_cl_loss = train_fc_loss = 0.0

        # Training is over
        for callback in self.callbacks:
            callback.on_train_end(self.history)
