import torch.autograd
import numpy as np
import torch

from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .utils import rampup, calculate_classification_metrics
from .basemodel import BaseModel


def detach_hidden_representations(hidden_representations: dict):
    for key, hidden_rep in hidden_representations.items():
        for layer, tensor in enumerate(hidden_rep):
            hidden_representations[key][layer] = tensor.detach().cpu()


class LadderNet(BaseModel):
    """ladder net implementation for time series classification
    """

    def __init__(self, backbone, backbone_dict,
                 loss_weights=None, callbacks=None):
        super(LadderNet, self).__init__(backbone=backbone,
                                        backbone_dict=backbone_dict,
                                        callbacks=callbacks)
        self.epochs_trained = 0
        self.num_layers = self.network.n_lateral_connections  # Also denoted as L

        # objective functions for both losses
        self.objective_sup = nn.CrossEntropyLoss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')

        self.loss_weights = loss_weights if loss_weights else [1.0] * self.num_layers

    def calculate_single_rc_loss(self, z: torch.Tensor,
                                 zhat: torch.Tensor,
                                 batch_mean: torch.Tensor, batch_std):
        """
        Normalize zhat with encoder µ and sd as described in
        Pezeshki et al. (2016) Eq. 18
        Args:
            zhat:
            batch_mean:
            batch_std:
        """
        assert batch_mean.ndim == 0, "Batch mean should be a scalar"
        assert batch_std.ndim == 0, "Batch std should be a scalar"
        assert z.shape == zhat.shape, "z and zhat should have same dimension"

        norm_zhats = (zhat - batch_mean) / batch_std
        return self.l2_loss(z, norm_zhats)

    def _calculate_total_loss(self, pred_y, true_y, zs, zhats, batch_mean,
                              batch_std, step=None,
                              rampup_length=None, max_w=5):
        """
        Internal method for calculating the laddernet loss. The loss function is
        described in Pezeshki et al. (2016) Eq. 4+5
        Args:
            pred_y:
                The predicted classes from the model.
            true_y:
                The true y from the dataset.
            zs:
                The z representations from the clean encoder
            zhats:
                The reconstructed representations z_hat from the decoder
            batch_mean:
                The batch means from the clean encoder
            batch_std:
                The batch standard deviations from the clean encoder
            step:
                The current training step
            ladders:
                The choice of what lateral connections to use for rc losses.
                Choose between: 'all', 'first' and 'last'
            rampup_length:
            max_w:
        Returns:
            The total loss and a list of all the reconstruction losses

        """
        idx_labelled = np.where(true_y.detach().cpu().numpy() != -1)[0]
        yhat = pred_y[idx_labelled]

        # Include a beta factor to slowly ramp up the unsupervised cost
        if rampup_length:
            beta = rampup(current=step,
                          rampup_length=np.min([step, rampup_length]))
            beta *= max_w
        else:
            beta = 1.0

        rc_losses = []  # Reconstruction losses

        encoder_idxes = range(self.num_layers)
        for encoder_idx in encoder_idxes:
            decoder_idx = self.num_layers - 1 - encoder_idx

            if self.loss_weights[encoder_idx] > 0:
                rc_losses.append(self.calculate_single_rc_loss(
                    zs[encoder_idx],
                    zhats[decoder_idx],
                    batch_mean[encoder_idx],
                    batch_std[encoder_idx]
                ))
            else:
                rc_losses.append(None)

        loss_sup = self.objective_sup(yhat, true_y[idx_labelled].long())

        loss = loss_sup
        for rc_loss_idx, loss_weight in enumerate(self.loss_weights):
            if loss_weight:
                loss += beta * loss_weight * rc_losses[rc_loss_idx]

        return loss, rc_losses

    def train(self,
              opt_dict,
              data_dict,
              model_params,
              exp_params,
              optimizer=optim.Adam):

        # This whole block is done here, but should ideally only be done
        # in LadderNet.__init__(). We need refactoring of the construction
        # of BaseModel to do this.
        if 'loss_weights' in model_params and model_params['loss_weights']:
            self.loss_weights = model_params['loss_weights']
        else:
            self.loss_weights = [1.0] * self.num_layers
        self.network.noise_sd = model_params['noise_sd']

        dataloader = data_dict['train_gen_l']
        optimizer = optimizer(self.network.parameters(), **opt_dict)

        if torch.cuda.is_available():
            self.network.to(torch.device('cuda'))

        if exp_params['lr_scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer=optimizer, eta_min=0.0,
                                          T_max=exp_params['n_steps'] * 1.2)
        else:
            scheduler = None

        for step in range(exp_params['n_steps']):
            self.network.train()  # Set the network in train mode

            for cb in self.callbacks:
                cb.on_train_batch_start()

            try:
                X, Y = next(train_gen_iter)
            except:
                train_gen_iter = iter(dataloader)
                X, Y = next(train_gen_iter)

            optimizer.zero_grad()
            if torch.cuda.is_available():
                X = X.to(torch.device('cuda'))
                Y = Y.to(torch.device('cuda'))

            yhat_all, hidden_reps = self.network(X, return_hidden_representations=True)

            loss, _ = self._calculate_total_loss(yhat_all, Y,
                                                 zs=hidden_reps['zs'],
                                                 zhats=hidden_reps['hat_zs'],
                                                 batch_mean=hidden_reps['batch_means'],
                                                 batch_std=hidden_reps['batch_std'],
                                                 step=step)
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
                metric = exp_params['early_stopping_metric']
                best_metric_performance = max(self.history[metric]) if len(self.history) > 0 else 0

                metrics = self.validate(step, {'lr': lr},
                              train_dataloader=data_dict['train_gen_val'],
                              val_dataloader=data_dict['val_gen'],
                              verbose=True)

                metric_performance = metrics[metric]

                if exp_params['early_stopping'] and metric_performance >= best_metric_performance:
                    print("step {} | New best model {}: {}, previous best model {}: {}".format(step, metric, metric_performance, metric, best_metric_performance))
                    self.save_checkpoint(step=step, verbose=True)

        # Training is over
        for callback in self.callbacks:
            callback.on_train_end(self.history)

    def predict(self, data_loader: DataLoader,
                internal_representations: bool = False):
        """
        Prediction functionality for the laddernet.
        Args:
            data_loader:
                A torch dataloader. This could be any dataloader such as
                train/val/test.
            internal_representations:
                Boolean to determine if the interal representations should be
                returned as well.

        Returns:
            The predicted y and the true y. Might also return
            hidden representations.
        """
        self.network.eval()

        all_y_pred = []
        all_y_true = []
        all_hidden_representations = []  # a list of a dict per batch with internal representations
        for batch_idx, (X, y_true_batch) in enumerate(data_loader):
            if torch.cuda.is_available():
                X = X.to(torch.device('cuda'))
                y_true_batch = y_true_batch.to(torch.device('cuda'))

            if internal_representations:
                y_pred, hidden_representations = self.network(X,
                                                              return_hidden_representations=True)
                detach_hidden_representations(hidden_representations)
                all_hidden_representations.append(hidden_representations)
            else:
                y_pred = self.network(X)

            all_y_pred.append(y_pred.detach().cpu().softmax(1))
            all_y_true.append(y_true_batch.detach().cpu())

        # Convert batches to single tensor
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y_true = torch.cat(all_y_true, dim=0)

        if internal_representations:
            return all_y_pred.numpy(), all_y_true.numpy(), all_hidden_representations
        else:
            # return tuple of three as this is the required input for the ABC's evaluate()
            return all_y_pred.numpy(), None, all_y_true.numpy()

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
        yhat_prob, y, hidden_representations = self.predict(data_loader, internal_representations=True)
        metrics = calculate_classification_metrics(yhat_prob, y)
        num_rc_losses = len(self.loss_weights)

        average_loss = []
        average_rc_losses = [[] for _ in range(num_rc_losses)]
        batch_idx = 0
        for hidden_rep_batch in hidden_representations:
            batch_size = hidden_rep_batch['zs'][0].shape[0]

            pred_y = yhat_prob[
                     batch_idx * batch_size:batch_idx + 1 * batch_size]
            true_y = y[batch_idx * batch_size:batch_idx + 1 * batch_size]

            # Make sure that y's are torch tensors
            pred_y = torch.Tensor(pred_y)
            true_y = torch.Tensor(true_y).int()

            loss, rc_losses = self._calculate_total_loss(
                pred_y=pred_y,
                true_y=true_y,
                zs=hidden_rep_batch['zs'],
                zhats=hidden_rep_batch['hat_zs'],
                batch_mean=hidden_rep_batch['batch_means'],
                batch_std=hidden_rep_batch['batch_std'],
                step=step
            )
            average_loss.append(loss.detach().cpu().numpy() / batch_size)
            for i in range(num_rc_losses):
                average_rc_loss = rc_losses[i].detach().cpu().numpy() / batch_size if rc_losses[i] else None
                average_rc_losses[i].append(average_rc_loss)

        metrics['loss'] = np.average(average_loss)

        for i, rc_l in enumerate(average_rc_losses):
            if any(rc_l):
                metrics[f'rc_loss_{i}'] = np.mean(rc_l)

        return metrics
