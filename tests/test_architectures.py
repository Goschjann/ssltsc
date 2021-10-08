from unittest import TestCase

from torch.utils.data import DataLoader

from dl4d.datasets.pamap import PAMAP2
from dl4d.datasets.cifar10 import Cifar10
from ssltsc.architectures.convlarge import ConvLarge, ConvLargeDecoder
from ssltsc.architectures.ladder import Ladder
from ssltsc.architectures.fcn import LadderFCN, LadderFCNDecoder

import os
import torch

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT_PATH, 'tests')


class TestImageArchitectures(TestCase):
    batch_size = 12

    cifar10_path = os.path.join(PROJECT_ROOT_PATH, "tests", "cifar10/")
    pamap2_path = os.path.join(PROJECT_ROOT_PATH, 'tests', 'pamap2/')

    cifar10_classes = 10
    cifar_channels = 3
    height = width = 32

    pamap2_classes = 19
    pamap2_channels = 6
    pamap2_length = 100

    pamap_dataloader = DataLoader(dataset=PAMAP2(root=DATA_ROOT, part='train'), batch_size=batch_size)
    one_pamap2_batch, _ = next(iter(pamap_dataloader))

    cifar_dataloader = DataLoader(dataset=Cifar10(root=DATA_ROOT, part='train'),
                                  batch_size=batch_size)
    one_cifar10_batch, _ = next(iter(cifar_dataloader))


    def setUp(self):
        if torch.cuda.is_available():
            self.one_cifar10_batch = self.one_cifar10_batch.to(torch.device('cuda'))
            self.one_pamap2_batch = self.one_pamap2_batch.to(torch.device('cuda'))

    def test_conv_large_forward_pass(self):
        self.assertEqual(self.one_cifar10_batch.shape, (self.batch_size, self.cifar_channels, self.height, self.width))

        # Make one forward pass and check that the outcome has the right shapes
        laddernet_architecture = ConvLarge(n_classes=self.cifar10_classes, channels=self.cifar_channels)

        out = laddernet_architecture(self.one_cifar10_batch)

        self.assertEqual((self.batch_size, 10), out.shape)

    def test_conv_large_ladder_forward_pass(self):
        self.assertEqual(self.one_cifar10_batch.shape, (self.batch_size, self.cifar_channels, self.height, self.width))

        # Make one forward pass and check that the outcome has the right shapes
        laddernet_architecture = Ladder(encoder_architecture=ConvLarge,
                                        decoder_architecture=ConvLargeDecoder,
                                        n_classes=self.cifar10_classes,
                                        channels=self.cifar_channels,
                                        )
        laddernet_architecture.train()
        out, hidden_reps = laddernet_architecture(self.one_cifar10_batch)

        self.assertEqual((self.batch_size, 10), out.shape)
        self.assertListEqual([13]*4, [len(h_reps) for h_reps in hidden_reps.values()])

        # Test the batch statistics shapes of the clean encoder hidden representations
        for m, std in zip(hidden_reps['batch_means'], hidden_reps['batch_std']):
            self.assertEqual((), tuple(m.shape))
            self.assertEqual((), tuple(std.shape))

        # Test the shapes of the hidden representations
        for z, z_hat in zip(hidden_reps['zs'], reversed(hidden_reps['hat_zs'])):
            self.assertEqual(z.shape, z_hat.shape)

    def test_conv_large_ladder_forward_pass_with_subset_of_ladders(self):

        # Counting from the top to bottom. This list say we need the top
        # ladder only ie. we only need one layer from the decoder
        ladders = [False] * 12 + [True]
        laddernet_architecture = Ladder(encoder_architecture=ConvLarge,
                                        decoder_architecture=ConvLargeDecoder,
                                        n_classes=self.cifar10_classes,
                                        channels=self.cifar_channels,
                                        ladders=ladders
                                        )
        laddernet_architecture.train()
        out, hidden_reps = laddernet_architecture(self.one_cifar10_batch)

        # Assert that we get correct length list back
        self.assertListEqual([13]*4, [len(h_reps) for h_reps in hidden_reps.values()])

        # Assert that everything other than the weighted loss is None. Here
        # we count from the top to button.
        self.assertListEqual([True] + [False] * 12, [hat_z is not None for hat_z in hidden_reps['hat_zs']])

    def test_fcn_forward_pass(self):

        self.assertEqual(self.one_pamap2_batch.shape, (self.batch_size, self.pamap2_channels, self.pamap2_length))

        # Make one forward pass and check that the outcome has the right shapes
        architecture = LadderFCN(channels=self.pamap2_channels, n_classes=self.pamap2_classes)

        out = architecture(self.one_pamap2_batch)

        self.assertEqual((self.batch_size, self.pamap2_classes), out.shape)

    def test_fcn_ladder_forward_pass(self):
        self.assertEqual(self.one_pamap2_batch.shape, (self.batch_size, self.pamap2_channels, self.pamap2_length))

        # Make one forward pass and check that the outcome has the right shapes
        laddernet_architecture = Ladder(encoder_architecture=LadderFCN,
                                        decoder_architecture=LadderFCNDecoder,
                                        n_classes=self.pamap2_classes,
                                        channels=self.pamap2_channels,
                                        length=self.pamap2_length)
        laddernet_architecture.train()
        out, hidden_reps = laddernet_architecture(self.one_pamap2_batch)

        self.assertEqual((self.batch_size, 19), out.shape)
        self.assertListEqual([5]*4, [len(h_reps) for h_reps in hidden_reps.values()])

        # Test the batch statistics shapes of the clean encoder hidden representations
        for m, std in zip(hidden_reps['batch_means'], hidden_reps['batch_std']):
            self.assertEqual((), tuple(m.shape))
            self.assertEqual((), tuple(std.shape))

        # Test the shapes of the hidden representations
        for z, z_hat in zip(hidden_reps['zs'], reversed(hidden_reps['hat_zs'])):
            self.assertEqual(z.shape, z_hat.shape)

    def test_fcn_ladder_forward_pass_with_subset_of_ladders(self):

        # Counting from the top to bottom. This list say we need the top
        # ladder only ie. we only need one layer from the decoder
        ladders = [False] * 4 + [True]
        laddernet_architecture = Ladder(encoder_architecture=LadderFCN,
                                        decoder_architecture=LadderFCNDecoder,
                                        n_classes=self.pamap2_classes,
                                        channels=self.pamap2_channels,
                                        length=self.pamap2_length,
                                        ladders=ladders
                                        )
        laddernet_architecture.train()
        out, hidden_reps = laddernet_architecture(self.one_pamap2_batch)

        # Assert that we get correct length list back
        self.assertListEqual([5]*4, [len(h_reps) for h_reps in hidden_reps.values()])

        # Assert that everything other than the weighted loss is None. Here
        # we count from the top to button.
        self.assertListEqual([True] + [False] * 4, [hat_z is not None for hat_z in hidden_reps['hat_zs']])