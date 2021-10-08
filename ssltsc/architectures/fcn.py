import torch
from torch import nn

from ssltsc.architectures.ladder_utils import Combinator1d, add_gaussian_noise

class LadderFCN(nn.Module):
    def __init__(self, n_classes, channels,
                 return_hidden_states=False, verbose=False):
        super(LadderFCN, self).__init__()
        self.n_classes = n_classes
        self.n_channels = channels

        self.n_lateral_connections = 3 + 2

        self.return_hidden_states = return_hidden_states
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_classes)

        # send the modules to cuda if gpu is present
        for m in self.modules():
            if torch.cuda.is_available():
                m.to(torch.device('cuda'))

    def forward(self, x, noise_sd=0.0):
        x = add_gaussian_noise(x, sd=noise_sd) if noise_sd else x

        z_pre1 = self.conv1(x)
        z1 = add_gaussian_noise(self.bn1(z_pre1), sd=noise_sd) if noise_sd else self.bn1(z_pre1)
        h1 = nn.functional.relu(z1)

        z_pre2 = self.conv2(h1)
        z2 = add_gaussian_noise(self.bn2(z_pre2), sd=noise_sd) if noise_sd else self.bn2(z_pre2)
        h2 = nn.functional.relu(z2)

        z_pre3 = self.conv3(h2)
        z3 = add_gaussian_noise(self.bn3(z_pre3), sd=noise_sd) if noise_sd else self.bn3(z_pre3)
        h3 = nn.functional.relu(z3)

        avg_pool = self.gap(h3)
        avg_pool = avg_pool.squeeze()
        out = self.fc(avg_pool)

        layers_z = [x, z1, z2, z3, out.unsqueeze(2)]  # The list of calculated z for each layer
        batch_means = [x.mean(), z_pre1.mean(), z_pre2.mean(), z_pre3.mean(), out.mean()]
        batch_std = [x.std(), z_pre1.std(), z_pre2.std(), z_pre3.std(), out.std()]

        if not self.return_hidden_states:
            return out

        if noise_sd:  # Then we will return z_hats and no mean and sd
            return out, layers_z

        return out, layers_z, batch_means, batch_std


class LadderFCNDecoder(nn.Module):
    def __init__(self, encoder, length: int = None, ladders: list = None):
        super(LadderFCNDecoder, self).__init__()
        self.ladders = ladders if ladders else [True] * 5
        self.n_classes = encoder.n_classes
        self.n_channels = encoder.n_channels

        self.combinator4 = Combinator1d(n_channels=self.n_classes, length=1)
        self.upsample = nn.Upsample(length)

        self.trans_conv3 = nn.ConvTranspose1d(self.n_classes, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.combinator3 = Combinator1d(n_channels=128, length=length)

        self.trans_conv2 = nn.ConvTranspose1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.combinator2 = Combinator1d(n_channels=256, length=length)

        self.trans_conv1 = nn.ConvTranspose1d(256, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.combinator1 = Combinator1d(n_channels=128, length=length)

        self.trans_conv0 = nn.ConvTranspose1d(128, self.n_channels, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm1d(num_features=self.n_channels)
        self.combinator0 = Combinator1d(n_channels=self.n_channels, length=length)

    def forward(self, layers_tilde_z: list):
        # Page 7 in Rasmus et al. (2015) states that "for the highest layer we
        # choose u=y_tilde". Ie. u is set to the softmax output from
        # the noisy encoder
        u = nn.functional.softmax(layers_tilde_z[-1], dim=1)  # Dims (bs, n_classes, 1)
        last_z_tilde = layers_tilde_z[-1]  # Dims (bs, n_classes, 1)
        assert u.shape == last_z_tilde.shape
        z_hat4 = self.combinator4(last_z_tilde, u)

        z_hat3 = self.trans_conv3(self.upsample(z_hat4))
        z_hat3 = self.bn3(z_hat3)
        assert layers_tilde_z[3].shape == z_hat3.shape, layers_tilde_z[3].shape
        z_hat3 = self.combinator3(layers_tilde_z[3], z_hat3)

        z_hat2 = self.trans_conv2(z_hat3)
        z_hat2 = self.bn2(z_hat2)
        assert layers_tilde_z[2].shape == z_hat2.shape
        z_hat2 = self.combinator2(layers_tilde_z[2], z_hat2)

        z_hat1 = self.trans_conv1(z_hat2)
        z_hat1 = self.bn1(z_hat1)
        assert layers_tilde_z[1].shape == z_hat1.shape
        z_hat1 = self.combinator1(layers_tilde_z[1], z_hat1)

        z_hat0 = self.trans_conv0(z_hat1)
        z_hat0 = self.bn0(z_hat0)
        assert layers_tilde_z[0].shape == z_hat0.shape
        z_hat0 = self.combinator0(layers_tilde_z[0], z_hat0)

        # This masking is done to adhere to the Laddernet convention.
        # Ideally we would only use the decoder when we need the ladder
        # connection. This could be done similarily to what is done in the
        # ConvLarge ladder net.
        z_hats = [z_hat4, z_hat3, z_hat2, z_hat1, z_hat0]
        last_needed_decoder = len(self.ladders) - self.ladders.index(True)
        z_hats_masked = [z_hat if decode_layer_idx < last_needed_decoder else None for decode_layer_idx, z_hat in enumerate(z_hats)]

        return z_hats_masked
