"""Code for Mean teacher model
"""
import torch.autograd
import numpy as np
import pandas as pd
import datetime

import torch
import os
import pdb
import datetime
import time
import cProfile
import torch.nn.functional as F

from itertools import cycle
from torch import nn, optim
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from sklearn.metrics import log_loss
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from .utils import ema_update, SigmoidScheduler, rampup
from .losses import softmax_mse_loss
from .basemodel import BaseModel
from ssltsc.models.utils import calculate_classification_metrics
from ssltsc.visualization import store_reliability

torch.set_default_dtype(torch.float32)

class MeanTeacher(BaseModel):
    """Mean Teacher model class

    Args:
        backbone: {nn.Module}

    """
    def __init__(self, backbone, backbone_dict, callbacks=None):
        super().__init__(backbone=backbone, backbone_dict=backbone_dict, callbacks=callbacks)
        self.student = backbone(**backbone_dict)
        self.teacher = backbone(**backbone_dict)
        self.network = None  # The network will be set throughout the train loop
        if torch.cuda.is_available():
            self.student.to(torch.device('cuda'))
            self.teacher.to(torch.device('cuda'))

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
        all_metrics = {}
        for submodel in ['student', 'teacher']:
            yhat_prob, yhat_logits, y = self.predict(data_loader, which=submodel)
            # calculate general metrics
            metrics = calculate_classification_metrics(yhat_prob, y)

            # calculate the 'originally' reduced loss
            logloss = nn.CrossEntropyLoss(weight=None, reduction='sum')

            metrics['loss'] = logloss(torch.tensor(yhat_logits), torch.tensor(y)).item() / len(data_loader.dataset)
            if submodel == 'teacher':
                all_metrics.update({k: v for k, v in metrics.items()})
            elif submodel == 'student':
                all_metrics.update({submodel + '_' + k: v for k, v in metrics.items()})

        return all_metrics

    def train(self,
              opt_dict,
              data_dict,
              model_params,
              exp_params,
              optimizer=optim.Adam):

        # objective function
        # ignore all -1 labels in the loss computation
        objective_sup = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
        optimizer = optimizer(self.student.parameters(), **opt_dict)

        if torch.cuda.is_available():
            self.student.to(torch.device('cuda'))
            self.teacher.to(torch.device('cuda'))

        # detach teacher from gradient flow
        for param in self.teacher.parameters():
            param.detach_()

        if exp_params['lr_scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer=optimizer,
                                          eta_min=0.0,
                                          T_max=exp_params['n_steps'] * 1.2)
        elif exp_params['lr_scheduler'] == 'sigmoid':
            scheduler = SigmoidScheduler(optimizer=optimizer,
                                         rampup_length=exp_params['rampup_length'])
        else:
            scheduler = None

        self.student.train()
        self.teacher.train()

        track_class_loss, track_cons_loss = 0, 0

        scaler = torch.cuda.amp.GradScaler()

        for step in range(exp_params['n_steps']):
            try:
                (X_stud, X_teach), Y = next(train_gen_iter)
            except:
                train_gen_iter = iter(data_dict['train_gen_l'])
                (X_stud, X_teach), Y = next(train_gen_iter)

            with torch.cuda.amp.autocast():
                if torch.cuda.is_available():
                    X_stud = X_stud.to(torch.device('cuda'))
                    X_teach = X_teach.to(torch.device('cuda'))
                    Y = Y.to(torch.device('cuda'))

                yhat_all_stud = self.student(X_stud)
                with torch.no_grad():
                    yhat_all_teach = self.teacher(X_teach)
                minibatch_size = len(Y)

                # objective_sup discards -1 labeled data
                loss_sup = objective_sup(yhat_all_stud, Y) / minibatch_size
                # combine losses
                beta = model_params['max_w'] * rampup(current=step,
                                                    rampup_length=model_params['rampup_length'])
                # mse over the predictions on all samples
                # softmax happens inside the loss
                # consistency loss
                loss_cons = beta * softmax_mse_loss(yhat_all_stud, yhat_all_teach) / minibatch_size
                loss = loss_sup + loss_cons
            track_class_loss += loss_sup.item()
            track_cons_loss += loss_cons.item()

            # update student
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # update teacher via exponential moving average
            ema_update(student=self.student,
                       teacher=self.teacher,
                       alpha=model_params['alpha_ema'],
                       verbose=False)

            if scheduler is not None:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
            else:
                lr = model_params['lr']
            # validation
            if step % exp_params['val_steps'] == 0 and step > 0:
                best_metric = 0.0 if len(self.history) == 0 else max(self.history[exp_params['early_stopping_metric']])
                metrics = self.validate(step=step,
                                hp_dict={'beta': beta, 'lr': lr},
                                train_dataloader=data_dict['train_gen_val'],
                                val_dataloader=data_dict['val_gen'])

                # early stopping
                self.network = self.teacher
                if exp_params['early_stopping'] and metrics[exp_params['early_stopping_metric']] > best_metric:
                    self.save_checkpoint(step=step, verbose=True)

        # Training is over
        for callback in self.callbacks:
            callback.on_train_end(self.history)

    def evaluate(self, data_loader: DataLoader, early_stopping: bool = False, which: str = 'teacher', plot_reliability: bool =False, model_name: str = 'meanteacher') -> dict:
        """
        This method will calculate the metrics on a given data_loader.
        Ie. the losses are not calculated in this method
        Args:
            data_loader:
                DataLoader to calculate metrics on
        Returns:
            A dict of metrics
        """
        self.network = self.teacher if which == 'teacher' else self.student
        if early_stopping:
            self.network = self.load_checkpoint()
        yhat_prob, _, y = self.predict(data_loader=data_loader)
        metrics = calculate_classification_metrics(yhat_prob, y)

        for callback in self.callbacks:
            callback.on_evaluation_end(metrics)

        if plot_reliability:
            store_reliability(y=y, yhat_prob=yhat_prob, model_name=model_name)

        return metrics

    def predict(self, data_loader: DataLoader, which='student'):
        self.network = self.student if which == 'student' else self.teacher
        # inherit predict method from ABC class
        preds = super().predict(data_loader=data_loader)
        self.network = None
        return preds

    def print_arch(self):
        print(self.student)


# timing decorator
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed

@timeit
def printer():
    for i in range(1000):
        a = i**2 - 98
    print('done')
