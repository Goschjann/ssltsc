import json
import os
import torch
import numpy as np
import pandas as pd
import tsfresh
from torch.utils import data

FC_PARAMETERS_2 = {
    'abs_energy': None,
    'absolute_sum_of_changes': None,
    'count_above_mean': None,
    'count_below_mean': None,
    'first_location_of_maximum': None,
    'first_location_of_minimum': None,
    'has_duplicate': None,
    'has_duplicate_max': None,
    'has_duplicate_min': None,
    'kurtosis': None,
    'last_location_of_maximum': None,
    'last_location_of_minimum': None,
    'length': None,
    'longest_strike_above_mean': None,
    'longest_strike_below_mean': None,
    'maximum': None,
    'minimum': None,
    'mean': None,
    'mean_abs_change': None,
    'mean_change': None,
    'mean_second_derivative_central': None,
    'median': None,
    'sample_entropy': None,
    'skewness': None,
    'standard_deviation': None,
    'sum_of_reoccurring_data_points': None,
    'sum_of_reoccurring_values': None,
    'sum_values': None,
    'variance': None,
    'variance_larger_than_standard_deviation': None,
}

np_type = 'float64'

def load_stats(path):
    with open('{}/stats.json'.format(path)) as f:
        return json.load(f)


def npy_to_pandas_df(arr):
    """
    For now required for feature extraction via tsfresh
    """
    num_obs = arr.shape[0]
    num_vars = arr.shape[1]
    ts_length = arr.shape[2]

    total_rows = num_obs * ts_length
    timeseries_arr = np.arange(ts_length) + 1

    time = np.tile(timeseries_arr, num_obs)
    obs_id = np.repeat([a for a in range(1, num_obs + 1)], ts_length)

    df_raw = pd.DataFrame({'obs_id': obs_id, 'time': time},
                          index=range(total_rows))

    variables = []
    for i in range(num_vars):
        variables.append('variable' + str(i + 1))

    df_input = pd.DataFrame(columns=variables, index=range(total_rows))

    for variable in range(len(variables)):
        df_input.iloc[:, variable] = arr[:, variable, :].reshape(total_rows, )

    df = df_raw.merge(df_input, left_index=True, right_index=True)

    return df


class TimeseriesDataset(data.Dataset):
    base_path = None

    def __init__(self, root, transform=None, target_transform=None,
                 task='classification',
                 horizon=None,
                 stride=None,
                 val_size=250,
                 test_size=2000,
                 normalize=False, standardize=False,
                 scale_overall=True, scale_channelwise=True):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        self.labelled_idxs = None

        self.transform = transform
        self.target_transform = target_transform

        self.task = task

        if task == 'forecast':
            assert 0.0 < stride < 1.0, 'Stride has to be in [0; 1]'
            assert 0.0 < horizon < 0.5, 'Horizon has to be in [0; 0.5]'
        self.horizon = horizon
        self.stride = stride

        # attributes for standardization/ normalization of the input data X
        # used in load_dataset()
        self.normalize = normalize
        self.standardize = standardize
        self.scale_overall = scale_overall
        self.scale_channelwise = scale_channelwise

        self.test_size = test_size # Absolute size of the test data set
        self.val_size = val_size # Absolute size of the validation data set

    @property
    def length(self):
        return self.x.shape[2]

    @property
    def nvariables(self):
        return self.x.shape[1]

    @property
    def nclasses(self):
        return len(np.unique(self.y))

    @property
    def stats(self):
        """
        The stats property will load a json file on every call. We might
        want to consider some caching.
        """
        return load_stats(path=os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform:
            x = torch.tensor(x)
            x = self.transform(x)

        if self.labelled_idxs and index not in self.labelled_idxs:
            y = -1

        return x, y

    def __len__(self):
        raise NotImplementedError

    def load_dataset(self, features=False, part='train'):

        if self.task == 'classification':
            return self._load_classification_dataset(features=features, part=part)
        elif self.task == 'forecast':
            return self._load_forecast_dataset(part)
        else:
            raise ValueError("The task can only be classification or forecast")

    def _load_classification_dataset(self, features, part='train'):
        path = os.path.join(self.root, self.base_folder)

        if features:
            x_path = os.path.join(path, 'X_{}_features.csv'.format(part))
            x = pd.read_csv(x_path).to_numpy()
        else:
            x_path = os.path.join(path, 'X_{}.npy'.format(part))
            x = np.load(file=x_path).astype(np_type)

        y = np.load(file=os.path.join(path, 'Y_{}.npy'.format(part))).astype('int')

        classes = np.unique(y)
        for idx in range(len(classes)):
            np.place(y, y == classes[idx], idx)
        y = y.astype(int)

        assert self.standardize + self.normalize < 2, "Either normalize OR standardize the data"

        if self.standardize:
            x, y = self.__standardize(x=x, y=y)
        if self.normalize and not self.standardize:
            x, y = self.__normalize(x=x, y=y)

        assert len(x) == len(y), "Length of X and y does not match"

        return x, y

    def _load_forecast_dataset(self, part='train'):
        path = os.path.join(self.root, self.base_folder)

        # stride = 0.3
        # horizon = 0.2

        x_forecast_file = "X_{}_forecast_s{}_h{}".format(part, self.stride, self.horizon)
        y_forecast_file = "Y_{}_forecast_s{}_h{}".format(part, self.stride, self.horizon)

        x_forecast_path = os.path.join(path, x_forecast_file)
        y_forecast_path = os.path.join(path, y_forecast_file)

        if os.path.exists(x_forecast_path) and os.path.exists(y_forecast_path):
            x = np.load(file=x_forecast_path).astype(np_type)
            y = np.load(file=y_forecast_path).astype(np_type)

            assert len(x) == len(y), "Length of X and y does not match"

            return x, y

        # Chunk the time series into forecasting tasks
        x_path = os.path.join(path, 'X_{}.npy'.format(part))
        x = np.load(file=x_path).astype(np_type)

        assert self.standardize + self.normalize < 2, "Either normalize OR standardize the data"

        if self.standardize:
            x, y = self.__standardize(x=x, y=y)
        if self.normalize and not self.standardize:
            x, y = self.__normalize(x=x, y=y)

        x, y = return_sliding_windows(x, stride_pct=self.stride, horizon_pct=self.horizon)
        np.save(file=x_forecast_path, arr=x)
        np.save(file=y_forecast_path, arr=y)

        return x, y

    def extract_features_from_npy(self):
        """
        Extract features from given path.
        Important function for the ssl ml baseline label prop models
        Uses the tsfresh feature extractor
        Is able to run on sktime and manual datasets
        Stores extracted features when finished

        Args:
            path: {string} name of the path were preprocessed data is stored
        Returns:
            X_train_features, X_test_features, X_val_features: {pd.DataFrame} extracted features
            from rawdata and stored in the same path as rawdata
        """
        path = os.path.join(self.root, self.base_folder)
        path_features = os.path.join(path, 'X_train_features.csv')
        path_features_test = os.path.join(path, 'X_test_features.csv')
        path_features_val = os.path.join(path, 'X_val_features.csv')

        try:
            X_train_arr = np.load(file=os.path.join(path, 'X_train.npy'))
            X_test_arr = np.load(file=os.path.join(path, 'X_test.npy'))
            X_val_arr = np.load(file=os.path.join(path, 'X_val.npy'))
        except AssertionError as error:
            print(error)
            print('X_train, X_test, X_val must be first saved '
                  'in numpy formatting in the given path')

        print('Extract features')

        # convert from numpy array to tsfresh compatible pd.DataFrame
        X_train_raw = npy_to_pandas_df(X_train_arr).fillna(0.0)
        X_test_raw = npy_to_pandas_df(X_test_arr).fillna(0.0)
        X_val_raw = npy_to_pandas_df(X_val_arr).fillna(0.0)

        X_train_features = tsfresh.extract_features(X_train_raw,
                                                    column_id='obs_id',
                                                    column_sort='time',
                                                    default_fc_parameters=FC_PARAMETERS_2)
        X_test_features = tsfresh.extract_features(X_test_raw,
                                                   column_id='obs_id',
                                                   column_sort='time',
                                                   default_fc_parameters=FC_PARAMETERS_2)
        X_val_features = tsfresh.extract_features(X_val_raw, column_id='obs_id',
                                                  column_sort='time',
                                                  default_fc_parameters=FC_PARAMETERS_2)
        X_train_features.to_csv(path_features, index=False)
        X_test_features.to_csv(path_features_test, index=False)
        X_val_features.to_csv(path_features_val, index=False)

    def save_stats(self, time_series: np.ndarray):
        """
        Collect stats from training data later used for scaling.
        """
        overall_dict = {'mean': time_series.mean(),
                        'std': time_series.std(),
                        'max': time_series.max(),
                        'min': time_series.min()}
        channel_dicts = []
        for channel in range(time_series.shape[1]):
            channel_dict = {'mean': time_series[:, channel, :].mean(),
                            'std': time_series[:, channel, :].std(),
                            'max': time_series[:, channel, :].max(),
                            'min': time_series[:, channel, :].min()}
            channel_dicts.append(channel_dict)
        final_dict = {i: channel_dicts[i] for i in range(time_series.shape[1])}
        final_dict['overall'] = overall_dict

        with open(os.path.join(self.root, self.base_folder, "stats.json"),
                  'w') as f:
            json.dump(final_dict, f, indent=True)

    def __standardize(self, x, y):
        """Z-normalize the data based on
        self.scale_overall:
            True: training data stats
            False: local stats
        self.scale_channelwise:
            True: per channel stats
            False: over all channels
        """

        if self.scale_overall:
            if self.scale_channelwise:
                temp_data = x
                for channel in range(x.shape[1]):
                    temp_data[:, channel, :] = (temp_data[:, channel, :] - self.stats[channel]['mean']) / \
                                               self.stats[channel]['std']
                return temp_data, y
            else:
                x = (x - self.stats['overall']['mean']) / self.stats['overall']['std']
                return x, y
        else:
            if self.scale_channelwise:
                temp_data = x
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        temp_data[i, j, :] = (temp_data[i, j, :] - temp_data[i, j, :].mean()) / temp_data[i, j, :].std()
                return temp_data, y
            else:
                temp_data = x
                for i in range(x.shape[0]):
                    temp_data[i, :, :] = (temp_data[i, :, :] - temp_data[i, :, :].mean()) / temp_data[i, :, :].std()
                return x, y

    def __normalize(self, x, y):
        """Min-max scale the data based on
        self.scale_overall:
            True: training data stats
            False: local stats
        self.scale_channelwise:
            True: per channel stats
            False: over all channels
        """
        if self.scale_overall:
            if self.scale_channelwise:
                temp_data = x
                for channel in range(x.shape[1]):
                    temp_data[:, channel, :] = (temp_data[:, channel, :] - self.stats[channel]['min']) / (
                                self.stats[channel]['max'] - self.stats[channel]['min'])
                return temp_data, y
            else:
                x = (x - self.stats['overall']['min']) / (self.stats['overall']['max'] - self.stats['overall']['min'])
                return x, y
        else:
            if self.scale_channelwise:
                temp_data = x
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        temp_data[i, j, :] = (temp_data[i, j, :] - temp_data[i, j, :].min()) / (
                                    temp_data[i, j, :].max() - temp_data[i, j, :].min())
                return temp_data, y
            else:
                temp_data = x
                for i in range(x.shape[0]):
                    temp_data[i, :, :] = (temp_data[i, :, :] - temp_data[i, :, :].min()) / (
                                temp_data[i, :, :].max() - temp_data[i, :, :].min())
                return temp_data, y


def return_sliding_windows(x: np.ndarray, stride_pct=0.2, horizon_pct=0.2):
    """
    A method for chunking a time series tensor (bs, channels, length) into
    a new time series tensor with a bigger number of time series of `horizon`
    length. This is

    Args:
        x: Time series tensor.
        stride_pct:
            The percentage of the input time series length to move
            the sliding window.
        horizon_pct:
            Determines the size of the output time series and the length of the
            time series to predict based on the input time series length.

    Returns:

    """
    is_batch = bool(len(x.shape) == 3)
    ts_length = x.shape[2] if is_batch else x.shape[1]

    if not is_batch:
        x = x.unsqueeze(0)

    xf = []
    yf = []

    stride = int(stride_pct * ts_length)  # Ie. how much do we move the window?
    horizon = int(horizon_pct * ts_length)  # Ie. how big a window?

    for x_start in range(0, ts_length, stride):
        x_end = x_start + horizon
        y_end = x_start + 2 * horizon

        if y_end <= ts_length:
            xf.append(x[:, :, x_start:x_end])
            yf.append(x[:, :, x_end:y_end])

    xf = np.concatenate(xf)
    yf = np.concatenate(yf)

    return xf, yf
