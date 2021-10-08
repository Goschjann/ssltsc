import shutil
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_and_extract_archive
from dl4d.timeseries import TimeseriesDataset


class PAMAP2(TimeseriesDataset):
    base_folder = 'pamap2'
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
    filename = "PAMAP2_Dataset.zip"

    seed = 1337  # Seed for train/test split
    _freq = 10  # Frequency for resampling in hz
    _length_in_sec = 10  # Length of each window in seconds
    _overlap = 0.8  # Overlap for data augmentation

    def __init__(self, root, part='train', task='classification',
                 transform=None, target_transform=None, download=True,
                 normalize=False, standardize=False,
                 features=False,
                 horizon=None,
                 stride=None,
                 val_size=250,
                 test_size=2000,
                 scale_overall=True, scale_channelwise=True):
        super(PAMAP2, self).__init__(root, transform=transform,
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
        extracted_path = os.path.join(self.root, 'PAMAP2_Dataset')

        if os.path.exists(final_path):
            return

        np.random.seed(self.seed)
        download_and_extract_archive(self.url, self.root,
                                     filename=self.filename)

        # colnames as provided by pamap dictionary
        imu_cols = ['temperature', 'acc_1_x', 'acc_1_y', 'acc_1_z',
                    'acc_2_x',
                    'acc_2_y', 'acc_2_y', 'gyro_x', 'gyro_y', 'gyro_z',
                    'magnet_x', 'magnet_y', 'magnet_z', 'orient_1',
                    'orient_2',
                    'orient_3', 'orient_4']

        columns = ['timestamp', 'act_id', 'heart_rate'] + \
                  ['hand_' + a for a in imu_cols] + \
                  ['chest_' + a for a in imu_cols] + \
                  ['ankle_' + a for a in imu_cols]

        resample_string = '{}S'.format(1 / self._freq)

        # columns of interest
        cols_int = ['act_id', 'hand_acc_1_x', 'hand_acc_1_y',
                    'hand_acc_1_z', 'hand_gyro_x', 'hand_gyro_y',
                    'hand_gyro_z']

        # store subjects and labels
        subject_list = []
        label_list = []

        for subject_idx in range(1, 10):
            print('##### Work on patient {}'.format(subject_idx))

            data_file = 'subject10{}.dat'.format(subject_idx)
            data_path = os.path.join(extracted_path, 'Protocol', data_file)
            data = pd.read_csv(data_path, sep='\s+', header=None)

            data.columns = columns
            # Amount of minutes for this subject
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
            data = data.set_index('timestamp')
            max_st = datetime.strptime(str(max(data.index)),
                                       '%Y-%m-%d %H:%M:%S.%f')
            min_st = datetime.strptime(str(min(data.index)),
                                       '%Y-%m-%d %H:%M:%S.%f')
            # Amount of total seconds
            secs = (max_st - min_st).total_seconds()
            st = min_st
            et = min_st

            # Initialize array with estimated amount of windows
            est_samples = int(secs / self._length_in_sec * (1 - self._overlap) ** (-1))
            df = np.empty(shape=(est_samples, len(cols_int) - 1, self._length_in_sec * self._freq))
            j = 0
            labels = []
            for col in cols_int:
                # Resampling to 10hz
                series = pd.Series(data[col], index=data.index)
                series = series.resample(resample_string).mean()

                # Segmenting into windows
                st = min_st
                et = min_st
                i = 0
                while True:
                    # Length of the window: 10 seconds
                    delta = timedelta(0, self._length_in_sec)
                    # Offset for the next window: 2 seconds for 80% args.overlap
                    offset = timedelta(0, (1 - self._overlap) * self._length_in_sec)
                    st = st + offset
                    et = st + delta
                    if et > max_st:
                        print('reached end, extracted {} windows'.format(i))
                        break
                    segment = series.between_time(start_time=st.time(),
                                                  end_time=et.time())
                    # Store time series or label
                    if col != 'act_id':
                        df[i, j, :] = segment.to_numpy()[
                                      :(self._length_in_sec * self._freq)]
                    else:
                        (v, c) = np.unique(segment.to_numpy(),
                                           return_counts=True)
                        idx = np.argmax(c)
                        labels.append(v[idx])
                    i += 1

                if col != 'act_id':
                    j += 1

            # Cut unneeded space in ndarray
            df = df[:i, :, :]

            subject_list.extend([subject_idx] * i)
            label_list.extend(labels)

            final_df = df if subject_idx == 1 else np.concatenate((final_df, df), axis=0)

        # fill nan's with 0.0
        X = np.nan_to_num(final_df)

        # store label and subject information
        meta_dict = {'subject': subject_list, 'label': label_list}
        meta_df = pd.DataFrame(meta_dict)

        Y = meta_df['label']
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
        np.save(file=os.path.join(final_path, 'Y_train.npy'), arr=Y_train.astype(np.float64))
        np.save(file=os.path.join(final_path, 'Y_test.npy'), arr=Y_test.astype(np.float64))
        np.save(file=os.path.join(final_path, 'Y_val.npy'), arr=Y_val.astype(np.float64))

        shutil.rmtree(extracted_path)
        self.save_stats(X_train)
        self.extract_features_from_npy()
