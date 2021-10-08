import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dl4d.timeseries import TimeseriesDataset


class SITS(TimeseriesDataset):
    base_folder = 'sits'
    url_train = "http://cloudstor.aarnet.edu.au/plus/s/pRLVtQyNhxDdCoM/download?path=%2FDataset%2FSITS_2006_NDVI_C%2FSITS1M_fold1&files=SITS1M_fold1_TRAIN.csv"
    url_test = "https://cloudstor.aarnet.edu.au/plus/s/pRLVtQyNhxDdCoM/download?path=%2FDataset%2FSITS_2006_NDVI_C%2FSITS1M_fold1&files=SITS1M_fold1_TEST.csv"
    filename_train = "SITS1M_fold1_TRAIN.csv"
    filename_test = "SITS1M_fold1_TEST.csv"

    num_obs = 100000
    min_support = 5000

    seed = 1337

    def __init__(self, root, part='train', task='classification',
                 transform=None, target_transform=None, download=True,
                 normalize=False, standardize=False,
                 features=False,
                 horizon=None,
                 stride=None,
                 val_size=250,
                 test_size=2000,
                 scale_overall=True, scale_channelwise=True):

        super(SITS, self).__init__(root, transform=transform,
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

        if not os.path.exists(final_path):
            os.mkdir(final_path)
        else:
            return

        np.random.seed(self.seed)

        # 2) Read in data
        df_raw_train = pd.read_csv(self.url_train, header=None)
        df_raw_test = pd.read_csv(self.url_test, header=None)

        # 3) Select random numbers of observations
        test_ratio = 0.1
        num_obs_test = round(self.num_obs * test_ratio)
        num_obs_train = round(self.num_obs * (1 - test_ratio))

        df_raw_test = df_raw_test.sample(num_obs_test)
        df_raw_train = df_raw_train.sample(num_obs_train)

        Y_train = np.asarray(df_raw_train.iloc[:, 0])
        Y_test = np.asarray(df_raw_test.iloc[:, 0])

        # Subset only classes with large support in the data
        large_classes = np.where(np.unique(Y_train, return_counts=True)[1] > self.min_support)[0] + 1
        idx_train = [idx for idx in range(len(Y_train)) if Y_train[idx] in large_classes]
        idx_test = [idx for idx in range(len(Y_test)) if Y_test[idx] in large_classes]

        Y_train = Y_train[idx_train]
        Y_test = Y_test[idx_test]

        print('Distribution of subsetted train and test')
        print(np.unique(Y_train, return_counts=True))
        print(np.unique(Y_test, return_counts=True))

        df_raw_train = df_raw_train.iloc[idx_train]
        df_raw_test = df_raw_test.iloc[idx_test]

        # Encode the labels to ints
        Y = pd.DataFrame(Y_train, columns=['label'])
        le = LabelEncoder()
        le.fit(Y['label'].values)
        Y_foo = le.transform(Y['label'].values)
        Y_train = pd.DataFrame(Y_foo.tolist(), columns=['label'])

        # Encode the labels to ints
        Y = pd.DataFrame(Y_test, columns=['label'])
        Y_foo = le.transform(Y['label'].values)
        Y_test = pd.DataFrame(Y_foo.tolist(), columns=['label'])

        df_raw_train = df_raw_train.iloc[:, 1:]
        df_raw_test = df_raw_test.iloc[:, 1:]

        # Reshape and normalize the data
        X_train = np.nan_to_num(df_raw_train.to_numpy())

        # Reshape to bcl format
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

        X_test = np.nan_to_num(df_raw_test.to_numpy())

        # Reshape to bcl format
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Split X_test data in X_test and X_val
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test,
                                                        test_size=self.val_size,
                                                        random_state=self.seed,
                                                        stratify=Y_test)

        np.save(file=os.path.join(final_path, 'X_train.npy'), arr=X_train)
        np.save(file=os.path.join(final_path, 'X_test.npy'), arr=X_test)
        np.save(file=os.path.join(final_path, 'X_val.npy'), arr=X_val)
        np.save(file=os.path.join(final_path, 'Y_train.npy'), arr=Y_train.astype(np.float64).squeeze(1))
        np.save(file=os.path.join(final_path, 'Y_test.npy'), arr=Y_test.astype(np.float64).squeeze(1))
        np.save(file=os.path.join(final_path, 'Y_val.npy'), arr=Y_val.astype(np.float64).squeeze(1))

        self.save_stats(X_train)
        self.extract_features_from_npy()
