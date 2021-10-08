import os
from unittest import TestCase

import numpy as np
import torch
from torch import Tensor

from torch.utils.data import DataLoader

from dl4d.datasets.pamap import PAMAP2
from dl4d.datasets.sits import SITS
from dl4d.datasets.ucr import CROP, ElectricDevices, FordB
from dl4d.datasets.wisdm import WISDM
from dl4d.sampler import SemiSupervisionSampler

from dl4d.transforms import TSTimeWarp, TSMagWarp, TSTimeNoise, TSMagNoise, TSMagScale

from dl4d.transforms import RandAugment, DuplicateTransform

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT_PATH, 'data')


class TestTimeseriesDataset(TestCase):

    # def test_dataset_lengths(self):
    #     """
    #     by testing the fixed length of the test data set we test
    #     the functionality of the dataloader irrespective of the
    #     validation set size
    #     """
    #     datasets = [
    #         (PAMAP2, 2000),
    #         (SITS, 7640),
    #         (WISDM, 2000),
    #         (CROP, 16550),
    #         (ElectricDevices, 7461),
    #         (FordB, 560)
    #     ]

    #     for dataset, length in datasets:
    #         # import pdb; pdb.set_trace()
    #         len_size = len(dataset(root=DATA_ROOT, part='test'))
    #         print(f'{len_size} dataset {length}')
    #         self.assertEqual(len(dataset(root=DATA_ROOT, part='test')), length)

    def test_classification(self):
        batch_size = 10
        pamap = PAMAP2(root=DATA_ROOT)
        dataloader = DataLoader(dataset=pamap, batch_size=10)

        x, y = next(iter(dataloader))

        self.assertEqual(x.shape, (batch_size, pamap.nvariables, pamap.length))
        self.assertEqual(y.shape, (batch_size,))

    def test_forecast(self):
        batch_size = 10

        with self.assertRaises(ValueError):
            PAMAP2(root=DATA_ROOT, task='regression')

        pamap = PAMAP2(root=DATA_ROOT, task='forecast', stride=0.5, horizon=0.2)
        dataloader = DataLoader(dataset=pamap, batch_size=10)

        x, y = next(iter(dataloader))

        self.assertEqual(x.shape[:2], (batch_size, pamap.nvariables))
        self.assertEqual(y.shape[:2], (batch_size, pamap.nvariables))


class TestSemiSupervisedDataLoading(TestCase):

    def test_semisupervised_loading(self):
        num_labels = 500
        batch_size = 12
        num_labels_in_batch = 5
        num_unlabelled_in_batch = batch_size - num_labels_in_batch

        pamap = PAMAP2(root=DATA_ROOT)
        sssampler = SemiSupervisionSampler(pamap, batch_size=batch_size,
                                           num_labels_in_batch=num_labels_in_batch,
                                           num_labels_in_dataset=num_labels)

        # The SemiSupervisionSampler will iterate over the unlabelled data
        # points once when iterating over the dataloader. We therefore
        # will end up with some residual unlabelled data points
        num_unlabelled_data_points = len(pamap) - num_labels
        num_batches = num_unlabelled_data_points // num_unlabelled_in_batch
        residual_unlabelled_data_points = num_unlabelled_data_points - num_batches * num_unlabelled_in_batch

        dataloader = DataLoader(dataset=pamap, batch_sampler=sssampler)

        num_unlabelled_count = 0
        for x, y in iter(dataloader):
            num_unlabelled_count += sum((y == -1).tolist())

        self.assertEqual(num_unlabelled_data_points,
                         num_unlabelled_count + residual_unlabelled_data_points)

    def test_semisupervised_loading_only_labelled(self):
        num_labels = 500
        batch_size = 10
        num_labels_in_batch = batch_size  # Then we should only loop over labelled data

        pamap = PAMAP2(root=DATA_ROOT)
        sssampler = SemiSupervisionSampler(pamap, batch_size=batch_size,
                                           num_labels_in_batch=num_labels_in_batch,
                                           num_labels_in_dataset=num_labels)

        dataloader = DataLoader(dataset=pamap, batch_sampler=sssampler)

        num_labelled_data_from_dl = 0
        for x, y in iter(dataloader):
            not_labelled_data_in_batch = sum((y == -1).tolist())
            self.assertEqual(not_labelled_data_in_batch, 0)

            num_labelled_data_from_dl += len(x)

        self.assertEqual(num_labelled_data_from_dl, num_labels)

    def test_semisupervised_loading_with_all_labels(self):
        # This would not make too much practical sense as then the user
        # would maybe simply not use the semisupervised sampler
        pamap = PAMAP2(root=DATA_ROOT)

        num_labels = len(pamap)  # Having all the datapoints labelled
        batch_size = 10
        num_labels_in_batch = batch_size  # Then we should only loop over labelled data

        num_batches = len(pamap) // batch_size
        last_batch_size = len(pamap) - num_batches * batch_size

        sssampler = SemiSupervisionSampler(pamap, batch_size=batch_size,
                                           num_labels_in_batch=num_labels_in_batch,
                                           num_labels_in_dataset=num_labels)

        dataloader = DataLoader(dataset=pamap, batch_sampler=sssampler)

        num_labelled_data_from_dl = 0
        for x, y in iter(dataloader):
            not_labelled_data_in_batch = sum((y == -1).tolist())
            self.assertEqual(not_labelled_data_in_batch, 0)

            num_labelled_data_from_dl += len(x)

        self.assertEqual(len(x), last_batch_size)
        self.assertEqual(num_labelled_data_from_dl, len(pamap))


class TestRandAugment(TestCase):
    pamap = PAMAP2(root=DATA_ROOT)
    data = pamap.x

    def setUp(self):
        self.transforms = [
            TSTimeWarp,
            TSMagWarp,
            TSTimeNoise,
            TSMagNoise,
            TSMagScale
        ]

    def test_single_transforms(self):
        data = torch.Tensor(self.data)

        # Try single transform of one example of type numpy array
        transformed_data = TSTimeWarp()(data[0])

        self.assertEqual(data[0].shape, transformed_data.shape)

        # Try a single transform on one example of type TSTensor
        transformed_data = TSTimeWarp()(data[0])

        self.assertEqual(data[0].shape, transformed_data.shape)

        # The transforms has changed the data
        self.assertFalse(np.array_equal(data, transformed_data))

        # Try transformation of multiple examples
        transformed_batch = TSTimeWarp()(data)

        self.assertEqual(data.shape, transformed_batch.shape)

        # The transforms has changed the batch
        self.assertFalse(np.array_equal(data.data, transformed_batch.data))

    def test_rand_augment_transforms(self):
        data = torch.Tensor(self.data)

        # Try a RandAugment transform on one example
        transformed_example = RandAugment(self.transforms)(data[0])
        self.assertEqual(transformed_example.shape, data[0].shape)
        self.assertFalse(np.array_equal(transformed_example.data, data[0].data))

        # Try a RandAugment transform of a batch
        transformed_batch = RandAugment(self.transforms)(data)
        self.assertEqual(transformed_batch.shape, data.shape)
        self.assertFalse(np.array_equal(transformed_example.data, data.data))

    def test_rand_augment_transform_on_torch_tensor(self):
        data = torch.from_numpy(self.data)

        self.assertTrue(type(data) == Tensor)

        # Try a RandAugment transform on one example
        transformed_example = RandAugment(self.transforms, num_transforms=4)(data[0])

        self.assertTrue(type(transformed_example) == Tensor)
        self.assertEqual(transformed_example.shape, self.data[0].shape)
        self.assertFalse(np.array_equal(transformed_example, data[0]))

        # Try a RandAugment transform of a batch
        transformed_batch = RandAugment(self.transforms, num_transforms=1)(data)

        self.assertTrue(type(transformed_batch) == Tensor)
        self.assertEqual(transformed_batch.shape, data.shape)
        self.assertFalse(np.array_equal(transformed_example, data))


class TestDuplicateTransform(TestCase):
    batch_size = 10
    num_labels_in_batch = 5
    num_labels = 500
    num_duplicates = 8  # in MixMatch this is aka K

    def test_duplicates_of_single_transform(self):
        # Create a transform
        tw_transform = TSTimeNoise()

        duplicate_transforms = DuplicateTransform(transform=tw_transform,
                                                  duplicates=self.num_duplicates)

        # Initialize the dataset with this new transform
        pamap = PAMAP2(root=DATA_ROOT, transform=duplicate_transforms)

        # Create a semisupervised sampling strategy
        sssampler = SemiSupervisionSampler(pamap, batch_size=self.batch_size,
                                           num_labels_in_batch=self.num_labels_in_batch,
                                           num_labels_in_dataset=self.num_labels)

        # Create the dataloader
        dataloader = DataLoader(dataset=pamap, batch_sampler=sssampler)

        list_of_duplicate_x_batches, y_batch = next(iter(dataloader))

        self.assertEqual(len(y_batch), self.batch_size)

        for x_batch in list_of_duplicate_x_batches:
            self.assertEqual(len(x_batch), self.batch_size)


        # Check two random transformed duplicates are not the same
        duplicate_idxs = np.random.choice(self.num_duplicates, 2, replace=False)

        first_duplicate = list_of_duplicate_x_batches[duplicate_idxs[0]]
        second_duplicate = list_of_duplicate_x_batches[duplicate_idxs[1]]

        batches_are_equal = (first_duplicate == second_duplicate).all()
        self.assertFalse(batches_are_equal)


    def test_duplicates_of_randaug_transform(self):
        # Create the rand aug transform
        rand_aug = RandAugment([
            TSTimeWarp,
            TSMagWarp,
            TSTimeNoise,
            TSMagNoise,
            TSMagScale
        ], num_transforms=3, magnitude=8)

        duplicate_transforms = DuplicateTransform(transform=rand_aug,
                                                  duplicates=self.num_duplicates)

        # Initialize the dataset with this new transform
        pamap = PAMAP2(root=DATA_ROOT, transform=duplicate_transforms)

        # Create a semisupervised sampling strategy
        sssampler = SemiSupervisionSampler(pamap, batch_size=self.batch_size,
                                           num_labels_in_batch=self.num_labels_in_batch,
                                           num_labels_in_dataset=self.num_labels)

        # Create the dataloader
        dataloader = DataLoader(dataset=pamap, batch_sampler=sssampler)

        list_of_duplicate_x_batches, y_batch = next(iter(dataloader))

        self.assertEqual(len(y_batch), self.batch_size)

        for x_batch in list_of_duplicate_x_batches:
            self.assertEqual(len(x_batch), self.batch_size)


        # Check two random transformed duplicates are not the same
        duplicate_idxs = np.random.choice(self.num_duplicates, 2, replace=False)

        first_duplicate = list_of_duplicate_x_batches[duplicate_idxs[0]]
        second_duplicate = list_of_duplicate_x_batches[duplicate_idxs[1]]

        batches_are_equal = (first_duplicate == second_duplicate).all()
        self.assertFalse(batches_are_equal)