import math

import torch
import torch.nn as nn

from torch.nn import init

torch.set_default_dtype(torch.float64)


class Affine(nn.Module):
    """
    This module implements the affine parameters gamma and beta seen in
    Eq. 10 in Pezeshki et al. (2016). It differs from the way affine
    is used in batchnorm out of the box of PyTorch.

    Pytorch affine      : y = bn(x)*gamma + beta
    Rasmus et al. (2015): y = gamma * (bn(x) + beta)
    """

    def __init__(self, n_channels, map_size):
        super(Affine, self).__init__()
        self.map_size = map_size
        self.n_channels = n_channels
        # initialize with no scaling (1) and shifting (0)
        # as well as in other implementations
        self.gamma = nn.Parameter(torch.Tensor(self.n_channels, self.map_size, self.map_size))
        self.beta = nn.Parameter(torch.Tensor(self.n_channels, self.map_size, self.map_size))

    def forward(self, x):
        out = self.gamma * (x + self.beta)
        return out

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.gamma, a=math.sqrt(5))
        init.kaiming_uniform_(self.beta, a=math.sqrt(5))


class Ladder(nn.Module):
    def __init__(self, encoder_architecture, decoder_architecture,
                 n_classes, channels, length=None, ladders=None,
                 noise_sd: float = 0.3, verbose=False):
        """

        Args:
            n_classes:
            channels:
            verbose:
        """
        super(Ladder, self).__init__()
        self.conv_net = encoder_architecture(n_classes=n_classes,
                                             channels=channels,
                                             verbose=verbose,
                                             return_hidden_states=True)
        self.noise_sd = noise_sd
        self.n_lateral_connections = self.conv_net.n_lateral_connections  # L
        self.ladders = ladders if ladders else [True] * self.n_lateral_connections
        assert self.n_lateral_connections == len(self.ladders)

        self.n_classes = n_classes

        self.decoder = decoder_architecture(self.conv_net, length=length,
                                            ladders=ladders)
        self.verbose = verbose
        self.first_pass = True

        # send the modules to cuda if gpu is present
        if torch.cuda.is_available():
            for m in self.modules():
                m.to(torch.device('cuda'))

    def clean_encoder(self, x):
        return self.conv_net(x)

    def noisy_encoder(self, x):
        return self.conv_net(x, noise_sd=self.noise_sd)

    def forward(self, x, return_hidden_representations=False):
        x_noise = x
        x_clean = x.clone()  # Detaching x

        # Always one pass through the clean encoder
        clean_logits, layers_z, batch_means, batch_std = self.clean_encoder(x_clean)

        if not self.training:  # Prediction mode uses the clean encoder
            y = nn.functional.softmax(clean_logits, dim=1)

        if return_hidden_representations or self.training:
            # When doing a pass for testing we do not need the noisy encoder
            noise_logits, layers_tilde_z = self.noisy_encoder(x_noise)
            layers_z_hat = self.decoder(layers_tilde_z)

            if self.training:  # Training mode uses noisy encoder
                y = nn.functional.softmax(noise_logits, dim=1)

            # We will return a dict of the hidden representations.
            # These are used when calculating the loss in a ladder model
            hidden_representations = {
                'zs': layers_z,
                'hat_zs': layers_z_hat,
                'batch_means': batch_means,
                'batch_std': batch_std
            }
            self.first_pass = False
            return y, hidden_representations

        self.first_pass = False
        return y

