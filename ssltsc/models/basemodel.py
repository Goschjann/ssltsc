"""Base model class
"""
import pdb
import numpy as np
import torch
import pandas as pd
import os
import tempfile

from torch import nn, optim
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from ssltsc.models.utils import calculate_classification_metrics
from ssltsc.visualization import store_reliability

class BaseModel(ABC):
    """ABC for all deep learning models
    """
    def __init__(self, backbone, backbone_dict, callbacks=None):
        self.backbone = backbone
        self.backbone_dict = backbone_dict
        self.history = pd.DataFrame()
        self.network = self.backbone(**self.backbone_dict)
        self.checkpoint_file = tempfile.NamedTemporaryFile()
        self.callbacks = callbacks if callbacks else []
        self.es_step = 0
        if torch.cuda.is_available():
            self.network.to(torch.device('cuda'))

    @abstractmethod
    def train(self,
              opt_dict,
              data_dict,
              model_params,
              exp_params,
              optimizer=optim.Adam):
        pass

    @abstractmethod
    def _validate_one_dataset(self):
        pass

    def validate(self, step, hp_dict, train_dataloader: DataLoader, val_dataloader: DataLoader = None,
                 verbose=True):
        """
        This method will validate two data_loaders (ie. train and val). It uses the _validate_one_dataset()
        for the actual metric and loss calculations.
        Args:
            step:
                The current step in the training process.
            hp_dict:
                dict of hyperparams we want to log each step
            train_dataloader:

            val_dataloader:
            verbose:
            plot_distribution:

        Returns:
            A history dict that contains metrics and losses and hparams.
        """
        history_data = hp_dict
        history_data['step'] = int(step)
        for part, dataloader in {"train": train_dataloader, "val": val_dataloader}.items():
            metrics = self._validate_one_dataset(dataloader, step)

            # Rename metrics for logging
            metrics = {part + "_" + metric: value for metric, value in metrics.items()}

            # Add data to history
            history_data.update(metrics)

        if not len(self.history.columns):  # If the columns names are not set then set them
            self.history = self.history.reindex(columns=history_data.keys())

        self.history = self.history.append(history_data, ignore_index=True)

        for callback in self.callbacks:
            callback.on_validation_end(step, history_data)

        return history_data


    def predict(self, data_loader):
        self.network.eval()

        yhat_list = []
        y_true_list = []

        for batch_idx, (X, Y) in enumerate(data_loader):
            if torch.cuda.is_available():
                X = X.to(torch.device('cuda'))
            Yhat = self.network(X).detach().cpu()
            # weird checking due to deep label prop algorithm which also outputs certainty weights
            Y_out = Y.detach().cpu() if len(Y)!= 2 else Y[0].detach().cpu()

            yhat_list.append(Yhat)
            y_true_list.append(Y_out)

        Yhat_logits = torch.cat(yhat_list, dim=0)
        Y_out = torch.cat(y_true_list, dim=0)
        Yhat = Yhat_logits.softmax(1)
        return Yhat.numpy(), Yhat_logits.numpy(), Y_out.numpy()

    def embed(self, gen):

        def embed_batch(model, x):
            """DOKUMENTATION!
            """
            for layer in model.children():
                if layer._get_name() == 'Linear':
                    break
                x = layer(x)
            return x

        self.network.eval()
        for batch_idx, (X, Y) in enumerate(gen):
            print('{}/{}'.format(batch_idx, len(gen)))
            if batch_idx == 0:
                V = embed_batch(model=self.network, x=X.to(torch.device('cuda'))).detach().cpu()
                # catch case where weights are added to train gen
                if len(Y) == 2:
                    Y_out = Y[0].detach().cpu()
                else:
                    Y_out = Y.detach().cpu()
            else:
                V = torch.cat((V, embed_batch(model=self.network, x=X.to(torch.device('cuda'))).detach().cpu()), dim=0)
                if len(Y) == 2:
                    Y_out = torch.cat((Y_out, Y[0].detach().cpu()), dim=0)
                else:
                    Y_out = torch.cat((Y_out, Y.detach().cpu()), dim=0)
        return V.numpy(), Y_out.numpy()

    def evaluate(self, data_loader: DataLoader, early_stopping: bool = False, plot_reliability: bool = False, model_name: str = None) -> dict:
        """
        This method will calculate the metrics on a given data_loader.
        Ie. the losses are not calculated in this method
        Args:
            data_loader:
                DataLoader to calculate metrics on
        Returns:
            A dict of metrics
        """
        if early_stopping:
            self.network = self.load_checkpoint()
        yhat_prob, _, y = self.predict(data_loader=data_loader)
        metrics = calculate_classification_metrics(yhat_prob, y)

        for callback in self.callbacks:
            callback.on_evaluation_end(metrics)

        if plot_reliability:
            store_reliability(y=y, yhat_prob=yhat_prob, model_name=model_name)

        return metrics

    def save_checkpoint(self, path: str = '', step: int = 1, verbose: bool = False):
        checkpoint = {'network': self.network,
                      'state_dict': self.network.state_dict(),
                      'step': step}
        # store checkpoint in random temp file (avoid issues training models in parallel)
        torch.save(checkpoint, self.checkpoint_file.name)
        print(f'Stored checkpoint at step {step}')

    def load_checkpoint(self, path: str = ''):
        checkpoint = torch.load(self.checkpoint_file.name)
        network = checkpoint['network']
        # store early stopping step
        self.es_step = checkpoint['step']
        print(f"Load Best Model from Step {self.es_step}")

        # overwrite network params with that of the checkpoint
        network.load_state_dict(checkpoint['state_dict'])
        for parameter in network.parameters():
            parameter.requires_grad = False
        network.eval()

        # clear checkpoint file
        self.checkpoint_file.close()
        return network