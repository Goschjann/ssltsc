import torchvision.transforms as transforms
import numpy as np
import pdb

from torch.utils.data import DataLoader, Dataset

from dl4d.datasets.pamap import PAMAP2
from dl4d.datasets.ucr import CROP, FordB, ElectricDevices
from dl4d.datasets.sits import SITS
from dl4d.datasets.wisdm import WISDM
from dl4d.datasets.cifar10 import Cifar10
from dl4d.datasets.svhn import SVHN
from dl4d.sampler import SemiSupervisionSampler, SupervisionSampler
from dl4d.transforms import TSRandomCrop, TSTimeWarp, TSMagWarp, TSMagScale, TSTimeNoise, \
    TSCutOut, TSMagNoise, RandAugment, DuplicateTransform, TransformFixMatch, RandAugmentMC

from functools import partial

#### Data Loader Functionalities

def get_inference_dataloader(dataset: Dataset, batch_size: int, num_workers: int):
    """
    This is a simple wrapper for a DataLoader. Everytime we load a dataset
    for inference we can use the DataLoader out of the box without drop_last
    and without shuffling.
    """
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=False,
                      drop_last=False)

def get_train_dataloader(dataset: Dataset, batch_size: int, num_workers: int,
                         num_labels_in_dataset: int,
                         drop_last: bool = False,
                         num_labels_in_batch: int = None,
                         sample_supervised: bool = False,
                         seed: int = 1337):
    """
    Connect the dataset with the sampling strategy to get a fully
    fletched data generator

    Args:
        dataset: An instantiated PyTorch Dataset
        batch_size:
        num_workers:
        num_labels_in_batch:
        num_labels_in_dataset:
        drop_last:
        seed:

    Returns:

    """
    if sample_supervised:
        sampler = SupervisionSampler(dataset,
                                     batch_size=batch_size,
                                     num_labels_in_dataset=num_labels_in_dataset,
                                     drop_last=drop_last, seed=seed)
    else:
        assert num_labels_in_batch, "To sample in semi supervised fashion we need to know num labels in batch"
        sampler = SemiSupervisionSampler(dataset, batch_size=batch_size,
                                         num_labels_in_batch=num_labels_in_batch,
                                         num_labels_in_dataset=num_labels_in_dataset,
                                         drop_last=drop_last, seed=seed)

    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=sampler,
                            num_workers=num_workers)

    return dataloader

def load_dataloaders(path: str,
                     dataset: str,
                     num_labels: int = 250,
                     model: str = None,
                     seed: int = 1337,
                     features: bool = False,
                     da_strategy: str = 'tf_rand_1',
                     fully_labeled: bool = False,
                     standardize: bool = False,
                     normalize: bool = False,
                     scale_overall: bool = False,
                     scale_channelwise: bool = False,
                     sample_supervised: bool = False,
                     K: int = 1,
                     batch_size: int = 64,
                     labeled_batch_size: int = 64,
                     inference_batch_size: int = 256,
                     num_workers: int = 10,
                     drop_last: bool = True,
                     N: int = 3,
                     magnitude: int = 1,
                     horizon: float = None,
                     stride: float = None,
                     val_size: int = 1000,
                     test_size: int = 2000,
                     **kwargs):

    dataset_classes = {
        'pamap2': PAMAP2,
        'fordb': FordB,
        'crop': CROP,
        'electricdevices': ElectricDevices,
        'sits': SITS,
        'wisdm': WISDM,
        'cifar10': Cifar10,
        'svhn': SVHN
    }


    if dataset not in dataset_classes:
        raise NotImplementedError()

    dataset_class = dataset_classes[dataset]

    # Select the data augmentation strategy
    if da_strategy == 'tf_rand_1' and dataset not in ['cifar10', 'svhn']:
        transform = tf_rand_1
        test_transform = None
    elif da_strategy == 'randaug' and dataset not in ['cifar10', 'svhn']:
        transform = make_randaug(N=N, magnitude=magnitude)
        test_transform = None
    elif da_strategy == 'fixmatch':
        sample_supervised = True
        if dataset not in ['cifar10', 'svhn']:
            transform = None # TODO
            test_transform = None
        else:
            # labelled transform
            transform = weak_cifar10_transformation
            # unlabelled transform
            unlabelled_transform = TransformFixMatch(
                weak=weak_cifar10_transformation,
                strong=strong_cifar10_transformation
            )
            # test transform
            test_transform = test_cifar10_transformation_1
    elif dataset == 'cifar10' and da_strategy is not False:
        transform = train_cifar10_transformation_2
        test_transform = test_cifar10_transformation_1
    elif dataset == 'svhn' and da_strategy is not False:
        transform = train_cifar10_transformation_5
        test_transform = test_cifar10_transformation_1
    else:
        transform = None
        test_transform = None

    # Load duplicates of x as required by MixMatch and MeanTeacher
    if K > 1:
        transform = DuplicateTransform(transform=transform, duplicates=K)

    train_ds = dataset_class(root=path, part='train', # labelled dataset with labelled transform and num_labels
                             task='classification',
                             features=features,
                             transform=transform, target_transform=None,
                             standardize=standardize, normalize=normalize,
                             scale_overall=scale_overall,
                             scale_channelwise=scale_channelwise,
                             val_size=val_size,
                             test_size=test_size)

    if model == 'selfsupervised':
        train_ds_forecast = dataset_class(root=path, part='train',
                                          task='forecast',
                                          horizon=horizon,
                                          stride=stride,
                                          features=features,
                                          transform=transform, target_transform=None,
                                          standardize=standardize, normalize=normalize,
                                          scale_overall=scale_overall,
                                          scale_channelwise=scale_channelwise,
                                          val_size=val_size,
                                          test_size=test_size)


    clean_train_ds = dataset_class(root=path, part='train',
                                   task='classification',
                                   features=features,
                                   transform=None, target_transform=None,
                                   standardize=standardize, normalize=normalize,
                                   scale_overall=scale_overall,
                                   scale_channelwise=scale_channelwise,
                                   val_size=val_size,
                                   test_size=test_size)

    val_ds = dataset_class(root=path, part='val',
                           task='classification',
                           features=features,
                           transform=test_transform, target_transform=None,
                           standardize=standardize, normalize=normalize,
                           scale_overall=scale_overall,
                           scale_channelwise=scale_channelwise,
                           val_size=val_size,
                           test_size=test_size)

    test_ds = dataset_class(root=path, part='test',
                            task='classification',
                            features=features,
                            transform=test_transform, target_transform=None,
                            standardize=standardize, normalize=normalize,
                            scale_overall=scale_overall,
                            scale_channelwise=scale_channelwise,
                            val_size=val_size,
                            test_size=test_size)

    ### Create dataloaders
    train_dl = get_train_dataloader(dataset=train_ds,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sample_supervised=sample_supervised,
                                    num_labels_in_batch=labeled_batch_size,
                                    num_labels_in_dataset=num_labels,
                                    drop_last=True,
                                    seed=seed)

    if model == 'fixmatch':
        train_ds_ul = dataset_class(root=path, part='train',
                                    task='classification',
                                    features=features,
                                    transform=unlabelled_transform, target_transform=None,
                                    standardize=standardize, normalize=normalize,
                                    scale_overall=scale_overall,
                                    scale_channelwise=scale_channelwise,
                                    val_size=val_size,
                                    test_size=test_size)
        ul_size = len(train_ds_ul)
        train_dl_u = get_train_dataloader(dataset=train_ds_ul,
                                          batch_size=batch_size * kwargs.get('mu', 7),
                                          num_workers=num_workers,
                                          sample_supervised=sample_supervised,
                                          num_labels_in_batch=labeled_batch_size,
                                          num_labels_in_dataset=ul_size,
                                          drop_last=True,
                                          seed=seed)
    else:
        train_dl_u = None

    if model == 'selfsupervised':
        # Basically supervised sampler of the forecast dataset
        train_dl_forecast = DataLoader(dataset=train_ds_forecast,
                                       batch_sampler=None,
                                       num_workers=num_workers,
                                       drop_last=True,
                                       shuffle=True,
                                       batch_size=batch_size)
    else:
        train_dl_forecast = None

    train_dl_val = get_train_dataloader(dataset=clean_train_ds,
                                        batch_size=inference_batch_size,  # for inference we can use a bigger batch size
                                        num_workers=num_workers,
                                        num_labels_in_dataset=num_labels,
                                        sample_supervised=True, # we only evaluate on the labelled training samples
                                        drop_last=False,
                                        seed=seed)

    test_dl = get_inference_dataloader(test_ds, inference_batch_size, num_workers)
    val_dl = get_inference_dataloader(val_ds, inference_batch_size, num_workers)

    return {'train_gen_l': train_dl,
            'train_gen_ul': train_dl_u,
            'train_gen_forecast': train_dl_forecast,
            'train_gen_val': train_dl_val,
            'test_gen': test_dl,
            'val_gen': val_dl,
            'train_data_l': train_ds,
            'test_data': test_ds,
            'val_data': val_ds}


#### Data Augmentation Functionalities

# time series

list_rand_1 = [TSTimeWarp,
               TSMagWarp,
               TSTimeNoise,
               TSMagNoise,
               TSMagScale,
               TSCutOut,
               TSRandomCrop]

tf_rand_1 = RandAugment(list_rand_1)

tf_rand_2 = RandAugment([TSMagScale,
                         partial(TSMagWarp, magnitude=9),
                         partial(TSCutOut, magnitude=9),
                         partial(TSTimeWarp, magnitude=8)])

# parametrized random augmentation:
def make_randaug(N=3, magnitude=1):
    list_randaug = [TSTimeWarp,
                    TSMagWarp,
                    TSTimeNoise,
                    TSMagNoise,
                    TSMagScale,
                    TSCutOut,
                    TSRandomCrop]
    return RandAugment(transformations=list_randaug,
                       num_transforms=N,
                       magnitude=magnitude)

# images

# cifar10 training data channel stats for normalization
channel_stats = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

# policy 1
train_cifar10_transformation_1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])

test_cifar10_transformation_1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats),
])

# policy 2
train_cifar10_transformation_2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.Pad(padding=4, padding_mode='reflect'),
    transforms.RandomCrop(size=32),
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])

test_cifar10_transformation_2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])

# policy 4
# randomcrop implements translation from Mean Teacher Paper
train_cifar10_transformation_4 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=2, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])

# policy 5
train_cifar10_transformation_5 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_cifar10_transformation_5 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


daug_proc = [TSTimeWarp,
            TSMagWarp,
            TSTimeNoise,
            TSMagScale]


# Fixmatch transforms

weak_cifar10_transformation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])

strong_cifar10_transformation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
    RandAugmentMC(n=2, m=10),
    transforms.ToTensor(),
    transforms.Normalize(**channel_stats)
])