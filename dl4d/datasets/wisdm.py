import shutil
import math
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_and_extract_archive
from dl4d.timeseries import TimeseriesDataset


class WISDM(TimeseriesDataset):
    base_folder = 'wisdm'
    url = 'http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz'
    filename = 'WISDM_ar_latest.tar.gz'

    seed = 1337
    overlap = 0.0
    length_sec = 4

    def __init__(self, root, part='train', task='classification',
                 transform=None, target_transform=None, download=True,
                 normalize=False, standardize=False, features=False,
                 horizon=None,
                 stride=None,
                 val_size=250,
                 test_size=2000,
                 scale_overall=True, scale_channelwise=True):

        super(WISDM, self).__init__(root, transform=transform,
                                    target_transform=target_transform,
                                    horizon=horizon,
                                    stride=stride,
                                    val_size=val_size,
                                    test_size=test_size,
                                    task=task, normalize=normalize, standardize=standardize,
                                    scale_overall=scale_overall, scale_channelwise=scale_channelwise)

        if download:
            self.download()

        self.x, self.y = self.load_dataset(part=part, features=features)
        self.test_size = test_size # Absolute size of the test data set
        self.val_size = val_size # Absolute size of the validation data set

    def __len__(self):
        return len(self.x)

    def download(self):
        final_path = os.path.join(self.root, self.base_folder)
        extracted_path = os.path.join(self.root, 'WISDM_ar_v1.1')

        if os.path.exists(final_path):
            return

        np.random.seed(self.seed)
        download_and_extract_archive(self.url, self.root,
                                     filename=self.filename)

        array_WISDM = np.loadtxt('{}/WISDM_ar_v1.1_raw.txt'.format(extracted_path), dtype=str)

        # Replace values
        timeseries_WISDM = [None] * array_WISDM.shape[0]
        for idx in range(array_WISDM.shape[0]):
            timeseries_WISDM[idx] = array_WISDM[idx].split(",")

        # Convert weird data format to pd.DataFrame, compatibel with tsfresh
        data = pd.DataFrame(timeseries_WISDM, columns=[
                            'ID', 'activity', 'timestamp',
                            'variable1', 'variable2', 'variable3',
                            'x1', 'x2', 'x3', 'x4', 'x5'])

        # Delete las columns ['x1' - 'x6'] as not relevant
        data = data.iloc[:, 0:6]
        data = data.dropna()

        # Last column has ; at the end of each numeric, replace with empty
        data.iloc[:, 5] = [i.replace(";", "") for i in data.iloc[:, 5].values]
        data['variable1'] = pd.to_numeric(data['variable1'])
        data['variable2'] = pd.to_numeric(data['variable2'])
        data['variable3'] = pd.to_numeric(data['variable3'])

        # Problem: time stamps are corrupted
        # Solution: no resampling etc, simply loop over them incl overlap

        lts = self.length_sec * 20
        timestep = math.ceil((1.0 - self.overlap) * lts)

        windows = []
        labels = []
        for i in range(0, len(data) - lts, timestep):
            x = data['variable1'].values[i: i + lts]
            y = data['variable2'].values[i: i + lts]
            z = data['variable3'].values[i: i + lts]
            windows.append([x, y, z])
            v, cnt = np.unique(data['activity'][i: i + lts], return_counts=True)
            idx = np.argmax(cnt)
            labels.append(v[idx])

        X = np.nan_to_num(np.asarray(windows, dtype=np.float64))

        # Encode the labels to ints
        Y = pd.DataFrame(labels, columns=['label'])
        le = LabelEncoder()
        le.fit(Y['label'].values)
        le.transform(Y['label'].values)
        Y_foo = le.transform(Y['label'].values)
        Y = pd.DataFrame(Y_foo.tolist(), columns=['label'])

        # Split X data in X_test and X_train
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=self.val_size + self.test_size,
                                                            random_state=self.seed,
                                                            stratify=Y)


        # Split X_test data in X_test and X_val
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test,
                                                        test_size=self.val_size,
                                                        random_state=self.seed,
                                                        stratify=Y_test)

        os.makedirs(final_path, exist_ok=True)

        np.save(file=os.path.join(final_path, 'X_train.npy'), arr=X_train)
        np.save(file=os.path.join(final_path, 'X_test.npy'), arr=X_test)
        np.save(file=os.path.join(final_path, 'X_val.npy'), arr=X_val)
        np.save(file=os.path.join(final_path, 'Y_train.npy'), arr=Y_train.astype(np.float64).squeeze(1))
        np.save(file=os.path.join(final_path, 'Y_test.npy'), arr=Y_test.astype(np.float64).squeeze(1))
        np.save(file=os.path.join(final_path, 'Y_val.npy'), arr=Y_val.astype(np.float64).squeeze(1))

        shutil.rmtree(extracted_path)
        self.save_stats(X_train)
        self.extract_features_from_npy()
