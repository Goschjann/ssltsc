"""
losses used in sslts
"""
import numpy as np
import torch
import pdb
import sklearn as sk

from torch.nn import functional as F
from numba import jit
from torch.autograd import Function


def mixup_cross_entropy(input_logits, targets):
    """cross entropy over 1hot-encoded (non binary) labels
    as required by mixup'ed labels
    """
    loss = - torch.mean(torch.sum(targets * torch.log(input_logits.softmax(1)), 1))
    return loss


def softmax_mse_loss(input_logits, target_logits):
    """takes softmax on both sides and returns MSE loss
        Returns the sum over all examples.

    Arguments:
        input_logits -- student logits
        target_logits -- 'true' teacher logits

    Returns:
        loss -- the mse consistency loss
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = input_logits.softmax(1)
    target_softmax = target_logits.softmax(1)
    num_classes = input_logits.size()[1]
    loss = F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes
    return loss


def entropy_loss(y_ul):
    """Entropy regularization for VAT
    """
    p = F.softmax(y_ul, dim=1)
    return -(p * F.log_softmax(y_ul, dim=1)).sum(dim=1).mean(dim=0)



##########
#   Soft dtw loss as proposed by cuturi et al 2017
#   implement by https://github.com/Sleepwalking/pytorch-softdtw
##########


@jit(nopython = True)
def compute_softdtw(D, gamma):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for k in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                r0 = -R[k, i - 1, j - 1] / gamma
                r1 = -R[k, i - 1, j] / gamma
                r2 = -R[k, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[k, i, j] = D[k, i - 1, j - 1] + softmin
    return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, : , -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
        return E[:, 1:N + 1, 1:M + 1]

class _SoftDTW(Function):
    @staticmethod
    def forward(ctx, D, gamma):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None


class SoftDTW(torch.nn.Module):
    def __init__(self, gamma=1.0, normalize=False):
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.func_dtw = _SoftDTW.apply

    def calc_distance_matrix(self, x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        dist = torch.pow(x - y, 2).sum(3)
        return dist

    def forward(self, x, y):
        assert len(x.shape) == len(y.shape)
        squeeze = False
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            squeeze = True
        if self.normalize:
            D_xy = self.calc_distance_matrix(x, y)
            out_xy = self.func_dtw(D_xy, self.gamma)
            D_xx = self.calc_distance_matrix(x, x)
            out_xx = self.func_dtw(D_xx, self.gamma)
            D_yy = self.calc_distance_matrix(y, y)
            out_yy = self.func_dtw(D_yy, self.gamma)
            # distance
            result = out_xy - (1 / 2) * (out_xx + out_yy)
        else:
            D_xy = self.calc_distance_matrix(x, y)
            out_xy = self.func_dtw(D_xy, self.gamma)
            # discrepancy
            result = out_xy
        return result.squeeze(0) if squeeze else result


def rbf_kernel_safe(X, Y=None, gamma=None):
    """Different rbf_kernel function which avoids returning of 0.0 distances
    Important for label prop/ spreading algorithms as semi-supervised baseline
    """
    X, Y = sk.metrics.pairwise.check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = sk.metrics.pairwise.euclidean_distances(X, Y, squared=True)
    K = K * (-1 * gamma)
    K -= K.max()
    np.exp(K, K)
    return K