import os
import pdb
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from dl4d.images import ImageDataset


class Cifar10(ImageDataset):
    base_folder = 'cifar10'
    seed = 1337
    val_size = 1000

    def __init__(self, root, part='train', task='classification',
                 features=False,
                 val_size=None,
                 test_size=None,
                 transform=None, target_transform=None, download=True,
                 normalize=False, standardize=False,
                 scale_overall=True, scale_channelwise=True):

        self.root = root
        if download:
            self.download()

        super(Cifar10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.x, self.y = self.load_dataset(part=part)

    def __len__(self):
        return len(self.x)

    def download(self):
        final_path = os.path.join(self.root, self.base_folder)
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        else:
            return

        np.random.seed(self.seed)

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root=final_path,
                                                train=True,
                                                download=True,
                                                transform=transform)
        testset = torchvision.datasets.CIFAR10(root=final_path,
                                               train=False,
                                               download=True,
                                               transform=transform)

        X_train = trainset.data.swapaxes(2, 3).swapaxes(1, 2)
        Y_train = trainset.targets
        X_test = testset.data.swapaxes(2, 3).swapaxes(1, 2)
        Y_test = testset.targets

        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test,
                                                        test_size=self.val_size,
                                                        random_state=self.seed,
                                                        stratify=Y_test)

        np.save(file=os.path.join(final_path, 'X_train.npy'), arr=X_train)
        np.save(file=os.path.join(final_path, 'X_test.npy'), arr=X_test)
        np.save(file=os.path.join(final_path, 'X_val.npy'), arr=X_val)
        np.save(file=os.path.join(final_path, 'Y_train.npy'), arr=Y_train)
        np.save(file=os.path.join(final_path, 'Y_test.npy'), arr=Y_test)
        np.save(file=os.path.join(final_path, 'Y_val.npy'), arr=Y_val)

        os.system('rm {}/cifar-10-python.tar.gz; rm -rf {}/cifar-10-batches-py'.format(final_path, final_path))