"""Implementation of deep label propagation
model by Iscen et al (2019)
"""

import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import faiss
import scipy
import scipy.stats
import pdb

from faiss import normalize_L2
from torch import nn, optim
from torch.utils import data
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR
from ssltsc.data.utils import SsltsData
from .utils import EarlyStopping
from .basemodel import BaseModel


# use float64 instead of the default float32
torch.set_default_dtype(torch.float64)

class Supervised(BaseModel):
    """Train backbone architecture supervised only as supervised baseline
    for ssl experiments
    """
    def __init__(self, backbone, backbone_dict):
        super().__init__(backbone=backbone, backbone_dict=backbone_dict)

    def train(self,
              n_steps,
              train_gen,
              opt_dict,
              n_pseudo_steps,
              val_gen=None,
              verbose=True,
              val_steps=100,
              optimizer=optim.Adam,
              lr_scheduler=None):
        """train the model for n_steps
        """
        # objective functions for both losses
        objective_sup = nn.CrossEntropyLoss(reduction='mean')
        objective_val = nn.CrossEntropyLoss(reduction='sum')

        optimizer = optimizer(self.network.parameters(), **opt_dict)

        if torch.cuda.is_available():
            self.network.to(torch.device('cuda'))

        if lr_scheduler:
            scheduler = CosineAnnealingLR(optimizer=optimizer,
                                          eta_min=0.0,
                                          T_max=(n_steps + n_pseudo_steps) * 1.2)
        else:
            scheduler = None

        for step in range(n_steps):
            self.network.train()
            try:
                X, Y = next(train_gen_iter)
            except:
                train_gen_iter = iter(train_gen)
                X, Y = next(train_gen_iter)

            optimizer.zero_grad()
            if torch.cuda.is_available():
                X = X.to(torch.device('cuda'))
                Y = Y.to(torch.device('cuda'))

            Yhat = self.network(X)
            loss = objective_sup(Yhat, Y)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']
            if step % val_steps == 0:
                self.validate(step=step, lr=lr, train_gen=train_gen, val_gen=val_gen)

    def train_pseudo(self,
                     n_steps,
                     n_pseudo_steps,
                     train_gen,
                     opt_dict,
                     nclasses,
                     train_gen_val,
                     train_rawdata_ul,
                     batch_size=64,
                     val_gen=None,
                     verbose=True,
                     val_steps=500,
                     optimizer=optim.SGD,
                     lr_scheduler=None,
                     stats=None,
                     standardize=False,
                     overall=True,
                     channelwise=True):
        """per sample weighted training of a model
        requires a dataloader with weigths
        e.g. used in dlp
        """
        # objective functions
        # now no reduction as we want to reweight it!
        # class weighted XE loss as in the paper
        objective_sup_val = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optimizer(self.network.parameters(), **opt_dict)

        if torch.cuda.is_available():
            self.network.to(torch.device('cuda'))

        if lr_scheduler:
            scheduler = CosineAnnealingLR(optimizer=optimizer,
                                          eta_min=0.0,
                                          T_max=(n_steps + n_pseudo_steps) * 1.2)
        else:
            scheduler = None

        self.network.train()
        ### this is the label propagation step
        # embed the data
        V, Y = self.embed(gen=train_gen)
        labeled_idx = np.where(Y != -1)[0]
        unlabeled_idx = np.where(Y == -1)[0]
        # update the labels and retrieve weights
        res_update = update_plabels(V=V,
                                    Y=Y,
                                    nclasses=nclasses,
                                    k=50,
                                    max_iter=20,
                                    labeled_idx=labeled_idx,
                                    unlabeled_idx=unlabeled_idx)
        plabels, weights, class_weights = res_update
        class_weights = torch.tensor(class_weights).to(torch.device('cuda'))
        objective_sup = nn.CrossEntropyLoss(weight=class_weights,
                                            reduction='none')
        # update the train generator accordingly
        updated_data = tuple([train_rawdata_ul[0], plabels])
        train_data_p = SsltsData(rawdata=updated_data,
                                    weights=weights,
                                    unlabeled=False,
                                    stats=stats,
                                    standardize=standardize,
                                    overall=overall,
                                    channelwise=channelwise)
        train_gen = data.DataLoader(dataset=train_data_p,
                                    batch_size=batch_size,
                                    shuffle=True)
        train_gen_pseudo_iter = iter(train_gen)

        for step in range(n_pseudo_steps):
            # run over the pseudo generator n_steps times
            # if empty, reinitialize it with updated weights
            try:
                X, (Y, w) = train_gen_pseudo_iter.next()
            except:
                print('get some juicy fresh embeddings at step {}'.format(step))
                ### this is the label propagation step
                # embed the data
                V, Y = self.embed(gen=train_gen)
                labeled_idx = np.where(Y != -1)[0]
                unlabeled_idx = np.where(Y == -1)[0]
                # update the labels and retrieve weights
                res_update = update_plabels(V=V, Y=Y,
                                            nclasses=nclasses,
                                            k=50, max_iter=20,
                                            labeled_idx=labeled_idx,
                                            unlabeled_idx=unlabeled_idx)
                plabels, weights, class_weights = res_update
                class_weights = torch.tensor(class_weights).to(torch.device('cuda'))
                objective_sup = nn.CrossEntropyLoss(weight=class_weights,
                                                    reduction='none')
                # update the train generator accordingly
                updated_data = tuple([train_rawdata_ul[0], plabels])
                train_data_p = SsltsData(rawdata=updated_data,
                                         weights=weights,
                                         unlabeled=False,
                                         stats=stats,
                                         standardize=standardize,
                                         overall=overall,
                                         channelwise=channelwise)
                train_gen = data.DataLoader(dataset=train_data_p,
                                            batch_size=batch_size,
                                            shuffle=True)
                train_gen_pseudo_iter = iter(train_gen)
                X, (Y, w) = train_gen_pseudo_iter.next()

            optimizer.zero_grad()
            if torch.cuda.is_available():
                X = X.to(torch.device('cuda'))
                Y = Y.to(torch.device('cuda'))
                w = w.to(torch.device('cuda'))

            Yhat = self.network(X)
            losses = objective_sup(Yhat, Y) * w
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']
            if step % val_steps == 0:
                self.validate(step=step, lr=lr, train_gen=train_gen, val_gen=val_gen)

def update_plabels(V, Y, labeled_idx, unlabeled_idx,
                   nclasses, k=50, max_iter=30, alpha=0.99):
    """update pseudo labels based on embedding from the net
    takes embedding V and true labels Y
    returns pseudo_labels, entropy_weights, class_weights
    """

    # normalize_L2 eats only doubles aka float32
    V = V.reshape(len(Y), -1).astype('float32')
    alpha = 0.99
    k = 50
    labels = Y
    # maxiter for Conjugate Gradient Solver
    max_iter = 20

    # kNN search for the graph
    d = V.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    # building the index takes some time
    index = faiss.GpuIndexFlatIP(res, d, flat_config)

    normalize_L2(V)
    index.add(V)
    N = V.shape[0]

    # D = similarity matrix
    # I = index matrix storing the indices of the k nn's per sample
    D, I = index.search(V, k + 1)

    # Create the graph
    # gamma parameter for manifold learning penalization as in eq 9
    gamma = 3
    D = D[:, 1:] ** gamma
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    # create sparse matrix with 50 nns -> 50*V.shape[0] elements != 0
    W = scipy.sparse.csr_matrix((D.flatten('F'),
                                 (row_idx_rep.flatten('F'), I.flatten('F'))),
                                shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class (eq 5 from the paper,
    # normalized with the class size) and apply label propagation
    Z = np.zeros((N, nclasses))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(nclasses):
        # a = np.where(labels[labeled_idx] == 0)[0]
        # cur_idx = [labeled_idx[b] for b in a]
        # CHECK WHAT HAPPENS HERE
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 / cur_idx.shape[0]
        # conjugate gradient from linalg
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter,
                                      atol=0)
        Z[:, i] = f

    # Handle numerical errors
    Z[Z < 0] = 0

    # Compute the weight for each instance based on the entropy
    # (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    probs_l1[probs_l1 < 0] = 0
    entropy = scipy.stats.entropy(probs_l1.T)
    weights = 1 - entropy / np.log(nclasses)
    weights = weights / np.max(weights)
    p_labels = np.argmax(probs_l1, 1)

    # Compute the accuracy of pseudolabels for statistical purposes
    correct_idx = (p_labels[labels != -1] == labels[labels != -1])
    print('pseudo label accuracy: {:0.5f}, mean entropy: {:0.5f}'.format(
        correct_idx.mean(), entropy.mean()))

    # fix the pseudo labels for the labeled data to its original label
    p_labels[labeled_idx] = labels[labeled_idx]
    weights[labeled_idx] = 1.0
    p_weights = weights.tolist()

    class_weights = []
    # Compute the class weights
    for i in range(nclasses):
        cur_idx = np.where(np.asarray(p_labels) == i)[0]
        class_weights.append(
            (float(labels.shape[0]) / nclasses) / cur_idx.size)

    return (p_labels, p_weights, class_weights)
