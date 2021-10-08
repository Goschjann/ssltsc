import itertools
import math

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Sampler, Dataset


class SemiSupervisionSampler(Sampler):
    """
    A sampler for loading a dataset in a semi supervised fashion.

    This sampler is inspired by the `TwoStreamBatchSampler`
    from the code for the Mean Teacher implementation by
    Curious AI.

    https://github.com/CuriousAI/mean-teacher/blob/546348ff863c998c26be4339021425df973b4a36/pytorch/mean_teacher/data.py#L105

    Args:
        dataset: A pytorch Dataset
        batch_size: The total batch size
        num_labels_in_batch: The number of labeled data in a batch
        num_labels_in_dataset: The number of labeled data in the dataset
        seed: Seed for the random unlabelling of the dataset

    Returns:
        defines the sampling procedure for the Dataloader later.
    """
    def __init__(self, dataset: Dataset, batch_size: int = 10,
                 num_labels_in_batch: int = 5,
                 num_labels_in_dataset: int = None,
                 seed: int = 1337,
                 drop_last: bool = False):

        assert batch_size < len(dataset), "Cannot load a batch bigger than the dataset"
        assert batch_size >= num_labels_in_batch, "Cannot have more labeled data in batch than batch size"
        assert num_labels_in_dataset <= len(dataset), "The number of labeled data in dataset must be smaller than the dataset size"

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.num_labels_in_batch = num_labels_in_batch
        self.num_labels_in_dataset = num_labels_in_dataset
        self.num_unlabelled_in_batch = batch_size - num_labels_in_batch

        dataset_idxs = range(len(dataset))

        self.unlabelled_idxs, self.labelled_idxs = make_unlabelling_split(
            dataset_idxs=dataset_idxs,
            y=dataset.y,
            num_labels_in_dataset=num_labels_in_dataset,
            seed=seed
        )

        dataset.labelled_idxs = self.labelled_idxs

        # If either we have a full labelled batch or a full labelled dataset
        # then we should never iterate over the unlabelled dataset
        full_labelled_batch = bool(batch_size == num_labels_in_batch)
        full_labelled_dataset = bool(num_labels_in_dataset == len(dataset))

        self.iterate_only_over_labelled = full_labelled_batch or full_labelled_dataset

    def __iter__(self):
        """
        Returns:
            A list of tuples where each tuple represents a batch and contains
            the idx for the datapoints in the given batch.
        """
        if self.iterate_only_over_labelled:
            labeled_iter = iterate_once(self.labelled_idxs)

            # This snippet is taken from the Pytorch BatchSampler.
            # It essentially loops over the indicies and fills up
            # batches. Once a batch is filled up then it is yielded
            batch = []
            for idx in labeled_iter:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch

        else:
            unlabeled_iter = iterate_once(self.unlabelled_idxs)
            labeled_iter = iterate_eternally(self.labelled_idxs)

            batches = zip(grouper(unlabeled_iter, self.num_unlabelled_in_batch),
                          grouper(labeled_iter, self.num_labels_in_batch))

            for (labeled_batch, unlabeled_batch) in batches:
                yield labeled_batch + unlabeled_batch


    def __len__(self):
        if self.iterate_only_over_labelled:
            if self.drop_last:
                return len(self.labelled_idxs) // self.batch_size

            # We will be doing ceil division because we do not want to drop_last
            return math.ceil(len(self.labelled_idxs) / self.batch_size)

        return len(self.unlabelled_idxs) // self.num_unlabelled_in_batch



class SupervisionSampler(Sampler):
    """
    A sampler for loading a dataset in a supervised fashion.

    Args:
        dataset: A pytorch Dataset
        batch_size: The total batch size
        num_labels_in_batch: The number of labeled data in a batch
        num_labels_in_dataset: The number of labeled data in the dataset
        seed: Seed for the random unlabelling of the dataset

    Returns:
        defines the sampling procedure for the Dataloader later.
    """
    def __init__(self, dataset: Dataset,
                 batch_size: int = 10,
                 num_labels_in_dataset: int = None,
                 seed: int = 1337,
                 drop_last: bool = False):

        assert batch_size < len(dataset), "Cannot load a batch bigger than the dataset"
        assert num_labels_in_dataset <= len(dataset), "The number of labeled data in dataset must be smaller than the dataset size"

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.num_labels_in_dataset = num_labels_in_dataset

        dataset_idxs = range(len(dataset))

        self.unlabelled_idxs, self.labelled_idxs = make_unlabelling_split(
            dataset_idxs=dataset_idxs,
            y=dataset.y,
            num_labels_in_dataset=num_labels_in_dataset,
            seed=seed
        )

        dataset.labelled_idxs = self.labelled_idxs

    def __iter__(self):
        """
        Returns:
            A list of tuples where each tuple represents a batch and contains
            the idx for the datapoints in the given batch.
        """
        labeled_iter = iterate_once(self.labelled_idxs)
        # This snippet is taken from the Pytorch BatchSampler.
        # It essentially loops over the indicies and fills up
        # batches. Once a batch is filled up then it is yielded
        batch = []
        for idx in labeled_iter:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.labelled_idxs) // self.batch_size

        # We will be doing ceil division because we do not want to drop_last
        return math.ceil(len(self.labelled_idxs) / self.batch_size)



def iterate_once(indicies):
    return np.random.permutation(indicies)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


def make_unlabelling_split(dataset_idxs, y, num_labels_in_dataset: int,
                           seed: int = 1337):
    """
    This function (ab)uses sklearns train_test_split() for stratified
    unlabelling
    Args:
        dataset_idxs:
        y:
        num_labels_in_dataset:
        seed:

    Returns:

    """

    if len(dataset_idxs) == num_labels_in_dataset:
        return [], dataset_idxs

    unlabelled_idxs, labelled_idxs, _, _ = train_test_split(
        dataset_idxs, y,
        test_size=num_labels_in_dataset,
        random_state=seed,
        stratify=y
    )

    return unlabelled_idxs, labelled_idxs
