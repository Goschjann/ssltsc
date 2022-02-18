"""Implementation of supervised baseline model
"""
import math
import mlflow
import os
import pdb
import shutil
import tempfile
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from .basemodel import BaseModel
from .utils import calculate_classification_metrics, get_cosine_schedule_with_warmup, \
        interleave, de_interleave, accuracy, AverageMeter

class Fixmatch(BaseModel):
    """Train backbone architecture supervised only as supervised baseline
    for ssl experiments
    """
    def __init__(self, backbone, backbone_dict, callbacks=None):
        super().__init__(backbone=backbone, backbone_dict=backbone_dict, callbacks=callbacks)

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file.name)
        network = checkpoint['model']
        self.es_step = checkpoint['epoch']

        if self.use_ema:
            network.load_state_dict(checkpoint['ema_state_dict'])
        else:
            network.load_state_dict(checkpoint['state_dict'])
        for parameter in network.parameters():
            parameter.requires_grad = False
        network.eval()

        self.checkpoint_file.close()
        return network

    def save_checkpoint(self, step):
        model_to_save = self.network.module if hasattr(self.network, "module") else self.network
        if self.use_ema:
            ema_to_save = self.ema_model.ema.module if hasattr(self.ema_model.ema, "module") else self.ema_model.ema
        
        checkpoint= {
            'step': step,
            'model': model_to_save,
            'state_dict': model_to_save.state_dict(),
            'ema_state_dict': ema_to_save.state_dict() if self.use_ema else None,
        }

        torch.save(checkpoint, self.checkpoint_file.name)
        print(f'Stored checkpoint at step {step}')

    def predict(self, data_loader):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.use_ema:
            eval_model = self.ema_model.ema
        else:
            eval_model = self.network
        
        yhat_list = []
        y_true_list = []
        with torch.no_grad():
            for _, (X, Y) in enumerate(data_loader):
                eval_model.eval()

                X = X.to(device)
                Yhat = eval_model(X).detach().cpu()
                Y_out = Y.detach().cpu()

                yhat_list.append(Yhat)
                y_true_list.append(Y_out)

        Yhat_logits = torch.cat(yhat_list, dim=0)
        Y_out = torch.cat(y_true_list, dim=0)
        Yhat = Yhat_logits.softmax(1)
        return Yhat.numpy(), Yhat_logits.numpy(), Y_out.numpy()

    def _validate_one_dataset(self, data_loader: DataLoader, step: int):
        yhat_prob, yhat_logits, y = self.predict(data_loader)
        metrics = calculate_classification_metrics(yhat_prob, y)

        metrics['loss'] = cross_entropy(torch.tensor(yhat_logits), torch.tensor(y).long()).item() / len(y)

        return metrics

    def train(self,
              opt_dict,
              data_dict,
              model_params,
              exp_params,
              optimizer=torch.optim.Adam):
        """train the model for n_steps
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.use_ema = model_params['use_ema']

        # optimizer and lr scheduler
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {
                'params': [p for n, p in self.network.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': model_params['weight_decay']
            },
            {
                'params': [p for n, p in self.network.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.SGD(grouped_parameters, lr=model_params['lr'], momentum=0.9, nesterov=True)
        if exp_params['lr_scheduler'] == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer, model_params['warmup_epochs'], exp_params['n_steps'])
        else:
            scheduler = None

        # ema model
        if self.use_ema:
            from .utils import ModelEMA
            self.ema_model = ModelEMA(self.network, model_params['ema_decay'], device)
        else:
            self.ema_model = None

        self.network.zero_grad()

        scaler = torch.cuda.amp.GradScaler()
        for step in tqdm(range(0, exp_params['n_steps'])):
            self.network.train()

            for cb in self.callbacks:
                cb.on_train_batch_start()
            
            try:
                inputs_x, targets_x = labelled_iter.next()
            except:
                labelled_iter = iter(data_dict['train_gen_l'])
                inputs_x, targets_x = labelled_iter.next()
            try:
                (inputs_u_w, inputs_u_s), _ = unlabelled_iter.next()
            except:
                unlabelled_iter = iter(data_dict['train_gen_ul'])
                (inputs_u_w, inputs_u_s), _ = unlabelled_iter.next()
            
            batch_size = inputs_x.shape[0]
            mu = inputs_u_w.shape[0] // batch_size

            with torch.cuda.amp.autocast():
                inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*mu+1).to(device)
                targets_x = targets_x.to(device)
                logits = self.network(inputs)
                logits = de_interleave(logits, 2*mu+1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                Lx = cross_entropy(logits_x, targets_x, reduction='mean')
                train_acc = accuracy(logits_x, targets_x, topk=(1,))[0]

                pseudo_label = torch.softmax(logits_u_w.detach()/model_params['temp'], dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(model_params['threshold']).float()
                Lu = (cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

                loss = Lx + model_params['lambda_u'] * Lu
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']

            if self.use_ema:
                self.ema_model.update(self.network)
            self.network.zero_grad()

            for cb in self.callbacks:
                cb.on_train_batch_end(step=step)
            
            if step % exp_params['val_steps'] == 0 and step > 0:
                best_metric = 0.0 if len(self.history) == 0 else max(self.history[exp_params['early_stopping_metric']])
                metrics = self.validate(step=step,
                                        hp_dict={'lr': lr},
                                        train_dataloader=data_dict['train_gen_val'],
                                        val_dataloader=data_dict['val_gen'])
                
                if exp_params['early_stopping'] and metrics[exp_params['early_stopping_metric']] > best_metric:
                    self.save_checkpoint(step=step)