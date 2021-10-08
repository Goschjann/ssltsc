import torch.nn as nn
import torch
import numpy as np


class Combinator(nn.Module):
    """
    The vanilla combinator function g() that combines vertical and
    lateral connections as explained in Pezeshki et al. (2016).
    The weights are initialized as described in Eq. 17
    and the g() is defined in Eq. 16.
    """

    def __init__(self, n_channels, length, data_type='2d'):
        super(Combinator, self).__init__()

        if data_type == '2d':
            zeros = torch.zeros(n_channels, length, length)
            ones = torch.ones(n_channels, length, length)
        elif data_type == '1d':
            zeros = torch.zeros(n_channels, length)
            ones = torch.ones(n_channels, length)
        else:
            raise ValueError

        self.b0 = nn.Parameter(zeros)
        self.w0z = nn.Parameter(ones)
        self.w0u = nn.Parameter(zeros)
        self.w0zu = nn.Parameter(ones)

        self.b1 = nn.Parameter(zeros)
        self.w1z = nn.Parameter(ones)
        self.w1u = nn.Parameter(zeros)
        self.w1zu = nn.Parameter(zeros)

        self.wsig = nn.Parameter(ones)

    def forward(self, z_tilde, ulplus1):
        assert z_tilde.shape == ulplus1.shape

        out = self.b0 + z_tilde.mul(self.w0z) + ulplus1.mul(self.w0u) \
              + z_tilde.mul(ulplus1.mul(self.w0zu)) \
              + self.wsig.mul(torch.sigmoid(self.b1 + z_tilde.mul(self.w1z)
                                            + ulplus1.mul(self.w1u)
                                            + z_tilde.mul(ulplus1.mul(self.w1zu))))
        return out


class Combinator2d(Combinator):
    def __init__(self, n_channels, length):
        super(Combinator2d, self).__init__(n_channels, length, data_type='2d')


class Combinator1d(Combinator):
    def __init__(self, n_channels, length):
        super(Combinator1d, self).__init__(n_channels, length, data_type='1d')


def add_gaussian_noise(x, sd=0.3):
    # We are only constructing a single random tensor that will be repeated
    # for each of the datapoints in the batch. This "hack" significantly
    # reduces speed during training.
    np_vec = np.random.normal(0.0, sd, x[0].size())
    noise = torch.Tensor(np_vec)

    # Alternatively we could generate a fully random tensor like this:
    # noise = torch.normal(0.0, 0.3, size=x.size())

    if torch.cuda.is_available():
        noise = noise.to(torch.device('cuda'))

    # Construct the noise tensor
    if len(x.shape) == 3:  # Then we have 1D data
        noise = noise.unsqueeze(0).repeat(x.size()[0], 1, 1)
    elif len(x.shape) == 4:  # Then we have 2D data
        noise = noise.unsqueeze(0).repeat(x.size()[0], 1, 1, 1)

    out = x + noise
    return out
