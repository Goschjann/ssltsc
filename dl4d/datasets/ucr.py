import shutil
import pdb
import os
import numpy as np
import tempfile
from sklearn.model_selection import train_test_split
from dl4d.timeseries import TimeseriesDataset
from urllib.request import urlretrieve
from pyunpack import Archive
from sktime.utils.data_io import load_from_tsfile_to_dataframe


class UCR(TimeseriesDataset):
    seed = 1337
    src_website = 'http://www.timeseriesclassification.com/Downloads'

    def __init__(self, root, dataset_name: str, part='train',
                 task='classification', transform=None,
                 target_transform=None, download=True,
                 normalize=False, standardize=False,
                 features=False,
                 horizon=None,
                 stride=None,
                 val_size=250,
                 test_size=2000,
                 scale_overall=True, scale_channelwise=True):

        super(UCR, self).__init__(root, transform=transform,
                                  target_transform=target_transform,
                                  task=task, normalize=normalize,
                                  horizon=horizon,
                                  stride=stride,
                                  standardize=standardize,
                                  scale_overall=scale_overall,
                                  val_size=val_size,
                                  test_size=test_size,
                                  scale_channelwise=scale_channelwise)
        if download:
            self._download(dataset_name)

        self.x, self.y = self.load_dataset(part=part, features=features)
        self.test_size = test_size # Absolute size of the test data set
        self.val_size = val_size # Absolute size of the validation data set

    def __len__(self):
        return len(self.x)

    def _download(self, dataset: str):
        final_path = os.path.join(self.root, self.base_folder)
        extracted_path = os.path.join(self.root, f'{dataset}_download')

        if not os.path.exists(final_path):
            os.makedirs(final_path)
            os.makedirs(extracted_path)
        else:
            return

        np.random.seed(self.seed)
        decompress_from_url(f'{self.src_website}/{dataset}.zip', target_dir=extracted_path)
        X_train_df, Y_train = load_from_tsfile_to_dataframe(f"{extracted_path}/{dataset}_TRAIN.ts")
        X_test_df, Y_test = load_from_tsfile_to_dataframe(f"{extracted_path}/{dataset}_TEST.ts")

        X_train_ = []
        X_test_ = []
        for i in range(X_train_df.shape[-1]):
            X_train_.append(stack_pad(X_train_df[f'dim_{i}']))
            X_test_.append(stack_pad(X_test_df[f'dim_{i}']))
        X_train = np.transpose(np.stack(X_train_, axis=-1), (0, 2, 1)).astype(np.float32)
        X_test = np.transpose(np.stack(X_test_, axis=-1), (0, 2, 1)).astype(np.float32)

        try:
            X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test,
                                                            test_size=self.val_size,
                                                            random_state=self.seed,
                                                            stratify=Y_test)
        except:
            print('X_test is too small, reduce X_val size to 400')
            X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test,
                                                            test_size=400,
                                                            random_state=self.seed,
                                                            stratify=Y_test)

        np.save(file=os.path.join(final_path, 'X_train.npy'), arr=X_train.astype(np.float32))
        np.save(file=os.path.join(final_path, 'X_test.npy'), arr=X_test.astype(np.float32))
        np.save(file=os.path.join(final_path, 'X_val.npy'), arr=X_val.astype(np.float32))
        np.save(file=os.path.join(final_path, 'Y_train.npy'), arr=Y_train.astype(np.float32))
        np.save(file=os.path.join(final_path, 'Y_test.npy'), arr=Y_test.astype(np.float32))
        np.save(file=os.path.join(final_path, 'Y_val.npy'), arr=Y_val.astype(np.float32))

        shutil.rmtree(extracted_path)
        self.save_stats(X_train.astype(np.float32))
        self.extract_features_from_npy()


class CROP(UCR):
    base_folder = 'crop'

    def __init__(self, root, part='train', task='classification',
                 transform=None, target_transform=None, download=True,
                 normalize=False, standardize=False,
                 features=False,
                 horizon=None,
                 stride=None,
                 val_size=250,
                 test_size=2000,
                 scale_overall=True, scale_channelwise=True):

        super().__init__(root, dataset_name='Crop', part=part,
                         transform=transform,
                         horizon=horizon,
                         stride=stride,
                         target_transform=target_transform,
                         task=task, normalize=normalize,
                         standardize=standardize,
                         scale_overall=scale_overall,
                         features=features,
                         val_size=val_size,
                         test_size=test_size,
                         scale_channelwise=scale_channelwise)


class FordB(UCR):
    base_folder = 'fordb'

    def __init__(self, root, part='train', task='classification',
                 transform=None, target_transform=None, download=True,
                 normalize=False, standardize=False,
                 features=False,
                 horizon=None,
                 stride=None,
                 val_size=250,
                 test_size=2000,
                 scale_overall=True, scale_channelwise=True):

        super().__init__(root, dataset_name='FordB', part=part,
                         transform=transform,
                         horizon=horizon,
                         stride=stride,
                         target_transform=target_transform,
                         task=task, normalize=normalize,
                         standardize=standardize,
                         scale_overall=scale_overall,
                         features=features,
                         val_size=val_size,
                         test_size=test_size,
                         scale_channelwise=scale_channelwise)


class ElectricDevices(UCR):
    base_folder = 'electricdevices'

    def __init__(self, root, part='train', task='classification',
                 transform=None, target_transform=None, download=True,
                 normalize=False, standardize=False,
                 features=False,
                 horizon=None,
                 stride=None,
                 val_size=250,
                 test_size=2000,
                 scale_overall=True, scale_channelwise=True):

        super().__init__(root, dataset_name='ElectricDevices',
                         part=part, transform=transform,
                         target_transform=target_transform,
                         horizon=horizon,
                         stride=stride,
                         task=task, normalize=normalize,
                         standardize=standardize,
                         scale_overall=scale_overall,
                         val_size=val_size,
                         test_size=test_size,
                         features=features,
                         scale_channelwise=scale_channelwise)

def decompress_from_url(url, target_dir=None, verbose=False):
    #Download
    try:
        fname = os.path.basename(url)
        tmpdir = tempfile.mkdtemp()
        local_comp_fname = os.path.join(tmpdir, fname)
        urlretrieve(url, local_comp_fname)
    except:
        shutil.rmtree(tmpdir)
        if verbose: sys.stderr.write("Could not download url. Please, check url.\n")

    #Decompress
    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        Archive(local_comp_fname).extractall(target_dir)
        shutil.rmtree(tmpdir)
        return target_dir
    except:
        shutil.rmtree(tmpdir)
        if verbose: sys.stderr.write("Could not decompress file, aborting.\n")
        return None

def stack_pad(l):
    def resize(row, size):
        new = np.array(row)
        new.resize(size)
        return new
    row_length = max(l, key=len).__len__()
    mat = np.array([resize(row, row_length) for row in l])
    return mat