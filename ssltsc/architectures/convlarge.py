import torch
from torch import nn

from ssltsc.architectures.ladder_utils import add_gaussian_noise, Combinator2d


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, kernel_size=3):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=1)

        # The easy implementation using standard affine. We could
        # use the affine from Eq. 10 in Pezeshki et al. (2016)
        self.bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

    def forward(self, x):
        z_pre = self.conv(x)
        z = self.bn(z_pre)
        return z, z_pre


class PoolLayer(nn.Module):
    def __init__(self, kernel_size, channels, padding=0):
        super(PoolLayer, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        z_pre = self.pool(x)
        z = self.bn(z_pre)
        return z, z_pre


class ConvLarge(nn.Module):
    def __init__(self, n_classes=3, channels=2, return_hidden_states=False, verbose=True):
        """

        Args:
            n_classes:
            channels:
            return_hidden_states:
                If the architecture is used in a Ladder architecture we need
                to return hidden states
            verbose:
        """
        super(ConvLarge, self).__init__()
        self.verbose = verbose
        self.n_classes = n_classes
        self.channels = channels
        self.return_hidden_states = return_hidden_states

        k = 3  # The default kernel size

        self.layers = nn.ModuleList()
        self.layers.append(ConvLayer(in_channels=self.channels, out_channels=96))  # Padding valid. Out shape (bs, 96, 30, 30)
        self.layers.append(ConvLayer(in_channels=96, out_channels=96, padding=k - 1))  # Padding full. Out shape (bs, 96, 32, 32)
        self.layers.append(ConvLayer(in_channels=96, out_channels=96, padding=k - 1))  # Padding full. Out shape (bs, 96, 34, 34)

        self.layers.append(PoolLayer(kernel_size=2, channels=96))  # Out shape (bs, 96, 17, 17)

        self.layers.append(ConvLayer(in_channels=96, out_channels=192))  # Padding valid. Out shape (bs, 192, 15, 15)
        self.layers.append(ConvLayer(in_channels=192, out_channels=192, padding=k - 1))  # Padding full. Out shape (bs, 192, 17, 17)
        self.layers.append(ConvLayer(in_channels=192, out_channels=192))  # Padding valid. Out shape (bs, 192, 15, 15)

        self.layers.append(PoolLayer(kernel_size=2, channels=192, padding=1))  # Original code uses downsize=2. Out shape (bs, 192, 8, 8)

        self.layers.append(ConvLayer(in_channels=192, out_channels=192))  # Padding valid. Out shape (bs, 192, 6, 6)
        self.layers.append(ConvLayer(in_channels=192, out_channels=192, kernel_size=1))  # Padding valid. Out shape (bs, 192, 6, 6)
        self.layers.append(ConvLayer(in_channels=192, out_channels=self.n_classes, kernel_size=1))  # Padding valid. Out shape (bs, 10, 6, 6)

        self.n_lateral_connections = len(self.layers) + 2  # 13
        # send the modules to cuda if gpu is present
        for m in self.modules():
            if torch.cuda.is_available():
                m.to(torch.device('cuda'))

    def forward(self, x, noise_sd=0.0):
        x = add_gaussian_noise(x, sd=noise_sd) if noise_sd else x
        h = x

        layers_z = [x]  # The list of calculated z for each layer
        batch_means = [x.mean()]
        batch_std = [x.std()]

        for l_idx, layer in enumerate(self.layers):
            z, z_pre = layer(h)  # h is set in the previous iteration

            z = add_gaussian_noise(z, sd=noise_sd) if noise_sd else z
            layers_z.append(z)

            h = nn.functional.leaky_relu(z)

            if not noise_sd:  # The mean and sd are only needed for clean pass
                batch_means.append(z_pre.mean())
                batch_std.append(z_pre.std())

        # Global average pool is the same as the mean of the last two dimensions
        global_avg_pool = h.mean([2, 3])

        layers_z.append(global_avg_pool.unsqueeze(2).unsqueeze(3))
        batch_means.append(global_avg_pool.mean())
        batch_std.append(global_avg_pool.std())

        assert len(self.layers) + 2 == len(layers_z)

        if not self.return_hidden_states:
            return global_avg_pool

        if noise_sd:  # Then we will return z_hats and no mean and sd
            return global_avg_pool, layers_z

        return global_avg_pool, layers_z, \
               batch_means, batch_std  # Ie. not the softmax but the logits


class DecodeLayer(nn.Module):
    def __init__(self, encode_layer=None, map_size=0):
        super(DecodeLayer, self).__init__()

        out_channels = encode_layer.conv.in_channels
        kernel_size = encode_layer.conv.kernel_size
        in_channels = encode_layer.conv.out_channels
        padding = encode_layer.conv.padding

        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding)

        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.combinator = Combinator2d(n_channels=out_channels, length=map_size)

    def forward(self, z_tilde, z_hat):
        u = self.trans_conv(z_hat)  # Eq. 11 in Pezeshki et al. (2015)
        u = self.bn(u)  # Eq 12-14 in Pezeshki et al. (2015)

        new_z_hat = self.combinator(z_tilde, u)  # Eq 15 in Pezeshki et al. (2015)

        return new_z_hat


class UpsampleLayer(nn.Module):
    """
    From Rasmus et al. (2015): "the downsampling of the pooling on the encoder
    side is compensated for by upsampling with copying on the decoder side."
    """
    def __init__(self, out_channels, feature_map_size):
        super(UpsampleLayer, self).__init__()

        self.upsample = nn.Upsample(feature_map_size)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.combinator = Combinator2d(n_channels=out_channels,
                                       length=feature_map_size)

    def forward(self, z_tilde, z_hat):
        u = self.upsample(z_hat)
        u = self.bn(u)

        new_z_hat = self.combinator(z_tilde, u)

        return new_z_hat


class ConvLargeDecoder(nn.Module):
    def __init__(self, encoder, length: int = None, ladders: list = None):
        """

        Args:
            encoder:
            length:
            ladders:
                A list of booleans that specifies whether the ladders should
                be used or not. The idx starts from bottom to top e.g.
                ladders[0] corresponds to the ladder between the first encoder
                layer and the last decoder layer. If ladder=None then all
                the lateral connections are used.
        """
        super(ConvLargeDecoder, self).__init__()
        self.num_layers = len(encoder.layers) + 2  # Also denoted L
        self.ladders = ladders if ladders else [True] * self.num_layers
        assert len(self.ladders) == self.num_layers

        self.decoder_layers = nn.ModuleList([
            Combinator2d(n_channels=10, length=1),
            UpsampleLayer(out_channels=10, feature_map_size=6),
            DecodeLayer(encode_layer=encoder.layers[-1], map_size=6),
            DecodeLayer(encode_layer=encoder.layers[-2], map_size=6),
            DecodeLayer(encode_layer=encoder.layers[-3], map_size=8),
            UpsampleLayer(out_channels=192, feature_map_size=15),
            DecodeLayer(encode_layer=encoder.layers[-5], map_size=17),
            DecodeLayer(encode_layer=encoder.layers[-6], map_size=15),
            DecodeLayer(encode_layer=encoder.layers[-7], map_size=17),
            UpsampleLayer(out_channels=96, feature_map_size=34),
            DecodeLayer(encode_layer=encoder.layers[-9], map_size=32),
            DecodeLayer(encode_layer=encoder.layers[-10], map_size=30),
            DecodeLayer(encode_layer=encoder.layers[-11], map_size=32)
        ])

    def forward(self, layers_tilde_z: list):

        # Page 7 in Rasmus et al. (2015) states that "for the highest layer we
        # choose u=y_tilde". Ie. u is set to the softmax output from
        # the noisy encoder
        u = nn.functional.softmax(layers_tilde_z[-1], dim=1)  # Dims (bs, n_classes, 1, 1)
        last_z_tilde = layers_tilde_z[-1]  # Dims (bs, n_classes, 1, 1)
        assert u.shape == last_z_tilde.shape

        z_hat = self.decoder_layers[0](last_z_tilde, u)  # l = 12
        layers_z_hat = [z_hat]

        last_needed_decoder = len(self.ladders) - self.ladders.index(True)

        for decode_layer_idx in range(1, self.num_layers):
            if decode_layer_idx >= last_needed_decoder:
                layers_z_hat.append(None)
                continue

            encode_layer_idx = - decode_layer_idx - 1
            tilde_z = layers_tilde_z[encode_layer_idx]

            z_hat = self.decoder_layers[decode_layer_idx](tilde_z, z_hat)
            layers_z_hat.append(z_hat)

        return layers_z_hat
