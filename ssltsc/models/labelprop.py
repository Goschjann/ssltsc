"""Baseline Label Propagation by Gahramani (2002)
"""
import random
import pdb
import numpy as np

from sklearn.semi_supervised import label_propagation
from sklearn.model_selection import train_test_split
from .losses import rbf_kernel_safe

class LabelProp():
    """Standard Label Propagation model based on sklearn implementation on
    feature extracted time series data.
        Args:
            num_labels: {int} amount of labeled data for training
            X_train: {pd.DataFrame} extracted features from training data
            y_train: {np.ndarray} all labels for the training data
    """
    def __init__(self, X_train, y_train, num_labels):
        self.num_labels = num_labels if num_labels < X_train.shape[0] else X_train.shape[0]
        self.X_train = X_train
        self.y_train = y_train
        self.__unlabel()

    def __unlabel(self):
        """unlabels the train data set using stratification
        """
        assert self.num_labels > 0.0, 'there have to be > 0.0 labeled samples!'
        X_ul, X_l, Y_ul, Y_l = train_test_split(self.X_train,
                                                self.y_train,
                                                test_size=int(self.num_labels),
                                                stratify=self.y_train)
        # 'unlabel' the data accordingly
        Y_ul = np.full(Y_ul.shape, -1)
        self.y_train = np.concatenate([Y_ul, Y_l], axis=0)
        self.X_train = np.concatenate([X_ul, X_l], axis=0)

    def train(self):
        """Trains the label propagation algorithm on the trainings data

        Args:
            None

        Returns:
            lp_model: {} fitted label propagation model
        """
        self.lp_model = label_propagation.LabelSpreading(kernel=rbf_kernel_safe,
                                                         gamma=0.25,
                                                         max_iter=1000)
        self.lp_model.fit(self.X_train, self.y_train)
        return self.lp_model

    def predict(self, X):
        """Returns the predictions on the test data based on the fitted label
        propagation model

        Args:
            None

        Returns:
        """
        self.predicted_probs = self.lp_model.predict_proba(X)
        self.predicted_labels = self.lp_model.predict(X)
        return (self.predicted_labels, self.predicted_probs)

    def plot_labelprop(self):
        pass
