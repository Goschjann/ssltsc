"""Utility functions for submodule 'models'
"""
from copy import deepcopy
import math
import numpy as np
import pandas as pd
import pdb
import torch

from sklearn.metrics import log_loss, roc_auc_score, f1_score
from uncertainty_metrics.numpy import ece

def ema_update(student, teacher, alpha=0.9, verbose=False):
    """Update a teacher model based on the exponential moving average
    of its weights and that of the current studen model.

    Controlled by alpha \\in [0, 1] with
        * alpha -> 1: teacher = past teacher
        * alpha -> 0: teacher = student, std SGD training
    Args:
        student: the student model
        teacher: the teacher
        alpha: ema alpha rate
        verbose: {bool} for checking: with alpha = 0.0 this should print True
                  only as weights from both models should be equal
    """
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        # alpha * theta'_t-1 + (1-a) * theta_t
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
        if verbose:
            print(teacher_param.data.equal(student_param.data))


class SigmoidScheduler:
    """"sigmoid rampup for learning rate as used in the
    mean teacher implement
    """
    def __init__(self, optimizer, rampup_length):
        self.optimizer = optimizer
        self.rampup_length = rampup_length
        self.counter = 0
        self.init_lr = optimizer.param_groups[0]['lr']
        self.last_lr = 0.0

    def step(self):
        self.optimizer.param_groups[0]['lr'] = self.init_lr * rampup(self.counter, self.rampup_length)
        self.counter += 1
        self.last_lr = self.optimizer.param_groups[0]['lr']

    def get_last_lr(self):
        return [self.last_lr]


def rampup(current, rampup_length):
    """sigmoid rampup
    """
    if current < rampup_length:
        p = max(0.0, float(current)) / float(rampup_length)
        p = 1.0 - p
        return float(np.exp(-p * p * 5.0))
    else:
        return 1.0


def linear_rampup(step, rampup_length=10):
    """linear rampup factor for the mixmatch model
    step = current step
    rampup_length = amount of steps till final weight
    """
    if rampup_length == 0:
        return 1.0
    else:
        return float(np.clip(step / rampup_length, 0, 1))


def calculate_classification_metrics(pred_prob_y, true_y) -> dict:
    """
    Wrapper to calculate all kinds of classification metrics
    which are then passed to the (mlflow) logger
    Args:
        pred_prob_y:
        true_y:
    Returns:
        A dictionary of metrics.
    """
    assert pred_prob_y[:, 0].shape == true_y.shape
    idx_labelled = np.where(true_y != -1)[0]
    pred_prob_y = pred_prob_y[idx_labelled]
    true_y = true_y[idx_labelled]
    yhat_hard = pred_prob_y.argmax(axis=1)

    # catch the binary case
    if pred_prob_y.shape[1] == 2:
        pred_prob_y = pred_prob_y[:, 1]
    metrics = {}
    # explicitly add list of possible labels in case of too small batch sizes
    # catch binary case as well
    labels = np.arange(pred_prob_y.shape[1]) if len(pred_prob_y.shape) > 1 else np.arange(2)
    metrics['ece'] = ece(labels=true_y, probs=pred_prob_y, num_bins=30)
    metrics['accuracy'] = sum(yhat_hard == true_y) / len(true_y)
    metrics['cross_entropy'] = log_loss(y_true=true_y, y_pred=pred_prob_y, labels=labels)
    metrics['weighted_auc'] = roc_auc_score(y_true=true_y, y_score=pred_prob_y, average='weighted', multi_class='ovr', labels=labels)
    metrics['macro_auc'] = roc_auc_score(y_true=true_y, y_score=pred_prob_y, average='macro', multi_class='ovo', labels=labels)
    metrics['macro_f1'] = f1_score(y_true=true_y, y_pred=yhat_hard, average='macro', labels=labels)
    metrics['micro_f1'] = f1_score(y_true=true_y, y_pred=yhat_hard, average='micro', labels=labels)
    metrics['weighted_f1'] = f1_score(y_true=true_y, y_pred=yhat_hard, average='weighted', labels=labels)

    return metrics


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


class ModelEMA(object):
    def __init__(self, model, ema_decay, device):
        self.ema = deepcopy(model)
        self.ema.to(device)
        self.ema.eval()
        self.decay = ema_decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
