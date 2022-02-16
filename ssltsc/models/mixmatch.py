"""
mixmatch model
"""
import numpy as np
import torch
import time
import pdb

from ssltsc import visualization
from tslearn.barycenters import softdtw_barycenter
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .utils import linear_rampup
from .basemodel import BaseModel
from .utils import calculate_classification_metrics

torch.set_default_dtype(torch.float32)


class MixMatch(BaseModel):
    """MixMatch Model class
    """

    def __init__(self, backbone, backbone_dict, callbacks=None):
        super().__init__(backbone=backbone, backbone_dict=backbone_dict, callbacks=callbacks)

    def _sharpen(self, x, T=0.5):
        """sharpen the predictions on the unlabeled data
        """
        temp = x ** (1 / T)
        return temp / temp.sum(dim=1, keepdim=True)

    def _indices_to_one_hot(self, data, nb_classes):
        """Convert an iterable of indices to one-hot encoded labels
        Args:
            data: {np.array} output array with the respective class
            nb_classes: {int} number of classes to one-hot encoded
        Returns:
            {np.array} a one-hot encoded array
        """
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def _mixup(self, x1, x2, y1, y2, alpha, dtw=False):
        """Mixup of two data points
        yields an interpolated mixture of both input samples
        """
        beta = np.random.beta(alpha, alpha)
        beta = max([beta, 1 - beta])
        if dtw:
            x = torch.empty(x1.shape)
            w1 = max([beta, 1 - beta])
            w = [w1, 1 - w1]
            for i in range(x.shape[0]):
                x[i, 0, :] = torch.tensor(softdtw_barycenter(X=[x1[i, 0, :].cpu(), x2[i, 0, :].cpu()],
                                                            weights=w)[:, 0])
            y = beta * y1 + (1 - beta) * y2
            return x.to(torch.device('cuda')), y
        else:
            x = beta * x1 + (1 - beta) * x2
            y = beta * y1 + (1 - beta) * y2
            return x, y

    def _interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def _interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self._interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

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
        """train the mixmatch model
        """
        optimizer = optimizer(self.network.parameters(), **opt_dict)
        self.n_classes = data_dict['train_gen_l'].dataset.nclasses

        # # detach ema_model from gradient flow
        # for param in self.ema_model.parameters():
        #     param.detach_()

        if torch.cuda.is_available():
            self.network.to(torch.device('cuda'))
            # self.ema_model.to(torch.device('cuda'))

        if exp_params['lr_scheduler']=='cosine':
            scheduler = CosineAnnealingLR(optimizer=optimizer,
                                          eta_min=0.0,
                                          T_max=exp_params['n_steps'] * 1.2)
        else:
            scheduler = None

        train_l_loss, train_ul_loss, train_mixmatch_loss = [0] * 3

        scaler = torch.cuda.amp.GradScaler()

        for step in range(exp_params['n_steps']):
            #start_time = datetime.datetime.now()
            optimizer.zero_grad()
            self.network.train()
            try:
                (x_1, x_2), y = next(train_gen_iter)
            except:
                train_gen_iter = iter(data_dict['train_gen_l'])
                (x_1, x_2), y = next(train_gen_iter)

            idx_unlabeled = np.where(y == -1)[0]
            idx_labeled = np.where(y != -1)[0]
            # sort labelled/ unlabelled data out
            # use only one augmented version of the labelled data set
            x_l = x_1[idx_labeled]
            y_l = y[idx_labeled]

            # double augmentation for the unlabelled data
            x_u1 = x_1[idx_unlabeled]
            x_u2 = x_2[idx_unlabeled]

            with torch.cuda.amp.autocast():
                if torch.cuda.is_available():
                    x_l = x_l.to(torch.device('cuda'))
                    y_l = y_l.to(torch.device('cuda'))
                    x_u1 = x_u1.to(torch.device('cuda'))
                    x_u2 = x_u2.to(torch.device('cuda'))

                # use notation from paper from now on
                xb = x_l
                # label guessing step for x_u
                ub = [x_u1, x_u2]
                # no autograd here, we take the guessed labels fo real bro
                with torch.no_grad():
                    # guess labels for usv samples via EMA in Mean Teacher
                    outputs = [self.network(u).softmax(1) for u in ub]
                    # take mean of the K augmented predictions
                    output = sum(i for i in outputs) / len(outputs)
                    # see lines 7 and 8 of the algorithm
                    qb = self._sharpen(output, model_params['T'])
                    Ux = torch.cat(ub, dim=0)
                    # replicate sharpened predictions qb K times
                    Uy = torch.cat([qb for _ in range(model_params['K'])], dim=0).detach()

                # shuffle step
                rand_idx = np.arange(len(xb) + len(Ux))
                np.random.shuffle(rand_idx)

                # one-hot encode y as this is required by mixup
                # guessed labels are already `one hot encoded`
                Y_enc = y_l.detach().cpu().numpy()
                Y_enc = self._indices_to_one_hot(Y_enc, self.n_classes)
                Y_enc = torch.from_numpy(Y_enc).to(torch.device('cuda'))

                # Wx and Wy are shuffled versions of the combined data set
                # to ensure in the mixup not the same X's are mixed up (sic!)
                Wx = torch.cat([xb, Ux], dim=0)[rand_idx.astype(int)]
                Wy = torch.cat([Y_enc, Uy], dim=0)
                Wy = Wy[rand_idx]

                # mix the labeled examples with the first n_labeled samples
                # of the shuffles W-matrix up
                X, xp = self._mixup(xb, Wx[:len(xb)], Y_enc, Wy[:len(xb)],
                                    model_params['alpha'], dtw=False)
                U, uq = self._mixup(Ux, Wx[len(xb):], Uy, Wy[len(xb):],
                                    model_params['alpha'], dtw=False)

                X_all = torch.cat([X, U], dim=0)

                ## INTERLEAVING
                # use interleaving from yuui1 to get valid batch norm stats
                n_labeled = xb.shape[0]
                X_all = list(torch.split(X_all, n_labeled))
                X_all = self._interleave(X_all, n_labeled)
                preds = [self.network(X_all[0]).softmax(1)]
                for inp in X_all[1:]:
                    preds.append(self.network(inp).softmax(1))
                # re-interleave the samples
                preds = self._interleave(preds, n_labeled)
                preds_x = preds[0]
                preds_u = torch.cat(preds[1:], dim=0)
                #print('train step took {}'.format(datetime.datetime.now() - start_time))

                if step % exp_params['val_steps'] == 0 and model_params['plot_mixup'] == 1:
                    visualization.plot_mixup(X1=xb[0], X2=Wx[0], X=X[0],
                                            Y1=Y_enc[0], Y2=Wy[0], Y=xp[0],
                                            comment='step{}'.format(step))

                # loss_labeled: XE
                # mean cross entropy labeled mixmatch labels p and model
                # predictions on labeled mixmatch input p_model
                # careful: preds are already softmaxed!
                # print('step {} target size {} pred size {}'.format(step, xp.shape[0], preds_x.shape[0]))
                loss_labeled = -torch.mean(torch.sum(xp * torch.log(preds_x), 1))

                # loss_unlabeled: mean squared error
                loss_unlabeled = torch.mean((preds_u - uq)**2)

                # combine losses
                ramp_factor = linear_rampup(step=step,
                                            rampup_length=model_params['rampup_length'])
                loss_unlabeled = (model_params['lambda_u'] * ramp_factor) * loss_unlabeled

                mixmatch_loss = loss_labeled + loss_unlabeled
                train_l_loss += loss_labeled.item()
                train_ul_loss += loss_unlabeled.item()
                train_mixmatch_loss += mixmatch_loss.item()
            # print('step {} mmloss {:.4f} lloss {:.4f} ulloss {:.4f}'.format(step, mixmatch_loss.item(), loss_labeled.item(), loss_unlabeled.item()))
            scaler.scale(mixmatch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # # update teacher via exponential moving average
            # ema_update(student=self.model,
            #            teacher=self.ema_model,
            #            alpha=model_params['alpha_ema'],
            #            verbose=False)

            if scheduler is not None:
                scheduler.step()

            # learning rate decay by 0.2 as mentioned in Real Eval Paper
            if step == 400000:
                optimizer.param_groups[0]['lr'] *= 0.8
            lr = optimizer.param_groups[0]['lr']

            # validation

            if step % exp_params['val_steps'] == 0 and step > 0:
                best_metric = 0.0 if len(self.history) == 0 else max(self.history[exp_params['early_stopping_metric']])
                metrics = self.validate(step=step,
                                        hp_dict={'lr': lr,
                                                 'lambda_u': model_params['lambda_u'],
                                                 'train_mixmatch_loss': train_mixmatch_loss / exp_params['val_steps'],
                                                 'train_l_loss': train_l_loss / exp_params['val_steps'],
                                                 'train_ul_loss': train_ul_loss / exp_params['val_steps']},
                                        train_dataloader=data_dict['train_gen_val'],
                                        val_dataloader=data_dict['val_gen'],
                                        verbose=True)

                # early stopping
                if exp_params['early_stopping'] and metrics[exp_params['early_stopping_metric']] > best_metric:
                    self.save_checkpoint(step=step, verbose=True)

                train_l_loss, train_ul_loss, train_mixmatch_loss = 0, 0, 0

        # Training is over
        for callback in self.callbacks:
            callback.on_train_end(self.history)

    def print_arch(self):
        print(self.network)
