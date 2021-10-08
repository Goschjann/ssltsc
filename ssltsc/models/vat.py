import numpy as np
import pandas as pd
import torch
import pdb
import mlflow
import datetime
import contextlib
import torch.nn.functional as F
import time

from ssltsc import visualization
from ssltsc.models import losses
from torch import nn, optim
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .basemodel import BaseModel

from .utils import calculate_classification_metrics


class VAT(BaseModel):
    """VAT Model class
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
        """train the vat model

        Arguments:
            train_gen_l {torch.DataLoader} -- labeled train generator
            train_gen_ul {torch.DataLoader} -- unlabeled train generator
            val_gen {torch.DataLoader} -- validation data
            opt_dict {dict} -- dictionary to parameterize optimizer

        Keyword Arguments:
            optimizer {torch.optim} -- optimizer available in torch.optim (default: {optim.Adam})
            n_steps {int} -- number of steps to train on (default: {500})
            val_steps {int} -- number of validation steps (default: {100})
            lr_scheduler {bool} -- use learning rate scheduler (default: {False})
            rampup_length {int} -- [description] (default: {60})
            xi {[type]} -- [description] (default: {1e-6})
            epsilon {int} -- [description] (default: {1})
            alpha {int} -- [description] (default: {1})
            method {str} -- [description] (default: {'vat'})
        """

        optimizer = optimizer(self.network.parameters(), **opt_dict)
        ce = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.network.to(torch.device('cuda'))

        if exp_params['lr_scheduler'] == 'cosine':
            scheduler = CosineAnnealingLr(optimizer=optimizer,
                                          eta_min=0.0,
                                          T_max=exp_params['n_steps'] * 1.2)
        # elif lr_scheduler == 'sigmoid':
        #     scheduler = SigmoidScheduler(optimizer=optimizer,
        #                                 rampup_length=rampup_length)
        else:
            scheduler = None

        # weight init for net
        self.network.train()

        train_ce_loss, train_vat_loss, train_ent_loss, train_tracking_loss = [0] * 4

        for step in range(exp_params['n_steps']):
            optimizer.zero_grad()

            try:
                x, y = next(train_gen_iter)
            except:
                train_gen_iter = iter(data_dict['train_gen_l'])
                x, y = next(train_gen_iter)

            idx_unlabeled = np.where(y == -1)[0]
            idx_labeled = np.where(y != -1)[0]
            x_l = x[idx_labeled]
            x_ul = x[idx_unlabeled]
            y_l = y[idx_labeled]# .squeeze(1)
            #print('data sorting took {}'.format(datetime.datetime.now() - start_time))

            if torch.cuda.is_available():
                x_l = x_l.to(torch.device('cuda'))
                y_l = y_l.to(torch.device('cuda'))
                x_ul = x_ul.to(torch.device('cuda'))
            # pdb.set_trace()
            logit = self.network(x_l)
            ce_loss = ce(logit, y_l)

            y_ul = self.network(x_ul)
            if step % 1000 == 0 and step > 0 and model_params['plot_adversarials']:
                plot_adv = True
            else:
                plot_adv = False

            vat_loss = self.vat_loss(x_ul=x_ul,
                                     y_ul=y_ul,
                                     xi=model_params['xi'],
                                     epsilon=model_params['epsilon'],
                                     method=model_params['method'],
                                     plot=plot_adv,
                                     step=step)

            if model_params['method'] == 'vatent':
                ent_loss = losses.entropy_loss(y_ul)
                loss = ce_loss + vat_loss + ent_loss
            else:
                loss = ce_loss + vat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log losses
            train_tracking_loss += loss.item()
            train_ce_loss += ce_loss.item()
            train_vat_loss += vat_loss.item()
            if model_params['method'] == 'vatent':
                train_ent_loss += ent_loss.item()

            if scheduler is not None:
                scheduler.step()

            if step == 400000:
                optimizer.param_groups[0]['lr'] *= 0.8
            lr = optimizer.param_groups[0]['lr']

            if step % exp_params['val_steps'] == 0 and step > 0:
                best_metric = 0.0 if len(self.history) == 0 else max(self.history[exp_params['early_stopping_metric']])
                metrics = self.validate(step=step,
                                        hp_dict={'lr': lr,
                                                'train_tracking_loss': train_tracking_loss / exp_params['val_steps'],
                                                'train_ce_loss': train_ce_loss / exp_params['val_steps'],
                                                'train_ent_loss': train_ent_loss / exp_params['val_steps'],
                                                'train_vat_loss': train_vat_loss / exp_params['val_steps']},
                                        train_dataloader=data_dict['train_gen_val'],
                                        val_dataloader=data_dict['val_gen'],
                                        verbose=True)

                # early stopping
                if exp_params['early_stopping'] and metrics[exp_params['early_stopping_metric']] > best_metric:
                    self.save_checkpoint(step=step, verbose=True)

                train_ce_loss, train_vat_loss, train_ent_loss, train_tracking_loss = [0] * 4
                self.network.train()

        # Training is over
        for callback in self.callbacks:
            callback.on_train_end(self.history)

    def vat_loss(self, x_ul, y_ul, xi, epsilon, method, plot=False, step=0):
        d = torch.cuda.DoubleTensor(x_ul.size()).normal_()
        d = xi * self._l2_normalize(d)
        d.requires_grad_()
        y_hat = self.network(x_ul + d)
        adv_distance = self.kl_divergence_with_logit(y_ul.detach(), y_hat)
        adv_distance.backward()

        d = d.grad.data.clone()
        self.network.zero_grad()

        d = self._l2_normalize(d)
        r_adv = epsilon * d
        y_hat = self.network(x_ul + r_adv.detach())
        lds = self.kl_divergence_with_logit(y_ul.detach(), y_hat)

        if plot:
            visualization.plot_vat_examples(x=x_ul, adv=r_adv, y=y_ul, path='', suffix=f'step_{step}')

        # if method == "vatent":
        #     entr_loss = self.entropy_loss(pred)
        #     lds = lds + entr_loss

        return lds

    def _l2_normalize(self, d):
        # normalization in torch directly
        # distinguish format for TS and images
        if len(d.shape) == 3:
            return d.div((torch.sqrt(torch.sum(d ** 2, dim=(1, 2))).view((-1, 1, 1)) + 1e-16))
        else:
            return d.div((torch.sqrt(torch.sum(d ** 2, dim=(1, 2, 3))).view((-1, 1, 1, 1)) + 1e-16))

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        logq = F.log_softmax(q_logit, dim=1)
        logp = F.log_softmax(p_logit, dim=1)

        qlogq = (q * logq).sum(dim=1).mean(dim=0)
        qlogp = (q * logp).sum(dim=1).mean(dim=0)

        return qlogq - qlogp

    def entropy_loss(self, ul_logit):
        p = F.softmax(ul_logit, dim=1)
        return -(p * F.log_softmax(ul_logit, dim=1)).sum(dim=1).mean(dim=0)

    @contextlib.contextmanager
    def _disable_tracking_bn_stats(self):

        def switch_attr(m):
            if hasattr(m, 'track_running_stats'):
                m.track_running_stats ^= True

        self.network.apply(switch_attr)
        yield
        self.network.apply(switch_attr)

    def plot_perturbation(self):
        pass

    def print_arch(self):
        print(self.network)