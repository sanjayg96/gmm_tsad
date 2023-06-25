"""
Implementation of GMM based time series anomaly detection.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import logging


class ADGMM:
    """
    Main class for ADGMM.

    Class to implement time series anomaly detection
    using Gaussian Mixture Models.

    """

    def __init__(
        self,
        n_components=1,
        remove_train_outliers=True,
        exponential_scale_factor=2,
        train_thresh=None,
        val_thresh=None,
        random_state=None,
    ):
        """
        Initialize parameters for the model.

        Args:

            n_componenets : int, default=1
                Number of GMM components fit for each bin.

            remove_train_outliers : bool, default=True
                Whether to remove the outliers in the training data before fitting the GMM model.
                Set to False if you do not want outliers to be removed before fitting.
                Set to True (default) to detect and remove outliers before fitting the model.

            exponential_scale_factor : int, values greater than or equal to 1, default=2
                Scaling factor applied on the predicted log probabilities to calculate scaled
                outlier score.
                The threshold is applied on the scaled outlier scores to classify the data points
                as either outlier or not.

            train_thresh : float, values in the closed interval [0, 10]
                Threshold used for replacing outliers detected in the training data before refitting.

            val_thresh : float, values in the closed interval [0, 10]
                Threshold used to classify a data point as outlier on the test data set.
                Greater the value, lesser the no. of detected outliers.

            random_state : int, default=0
                Set random seed for result reproducibility.

        Returns:

            ADGMM object

        """

        if (remove_train_outliers == True) and (
            train_thresh == None or train_thresh < 0
        ):
            raise ValueError(
                """ A float value between 1 to 10 must be specified for the argument 'train_thresh'
                if 'remove_train_outliers' is set to True."""
            )

        if val_thresh == None:
            raise ValueError(
                "A float value between 1 to 10 must be specified for the argument 'val_thresh'."
            )

        self.n_components = n_components
        self.remove_train_outliers = remove_train_outliers
        self.exponential_scale_factor = exponential_scale_factor
        self.train_thresh = train_thresh
        self.val_thresh = val_thresh
        self.random_state = random_state

    def build(self, series):
        """
        Method to fit GMM model on a single bin.

        Args:

            series : pandas Series.
                The data belonging to a particular bin.

        Returns:

            GMM model fit to the bin data.
        """

        if self.remove_train_outliers:
            gmm_1 = GaussianMixture(
                n_components=self.n_components,
                covariance_type="spherical",
                random_state=self.random_state,
            )

            gmm_1.fit(series.values.reshape(-1, 1))

            df = pd.DataFrame()
            df["values"] = series
            # get log(p(x)) and raise to the power 2*f, where f is the exponential_scale_factor
            df["OSx"] = gmm_1.score_samples(series.values.reshape(-1, 1)) ** (
                2 * self.exponential_scale_factor
            )
            # scale the values from 0 to 10
            df["ScOSx"] = (
                MinMaxScaler().fit_transform(df["OSx"].values.reshape(-1, 1)) * 10
            )
            df["ScOSx"] = round(df["ScOSx"], 2)

            # Detect outliers
            replace_idx = df[df["ScOSx"] >= self.train_thresh].index.values

            weights = gmm_1.weights_
            means = gmm_1.means_

            # Replace outliers with the mean of the Gaussian with the highest weight
            df["values"].loc[replace_idx] = means[np.argmax(weights)][0]

            gmm_refit = GaussianMixture(
                n_components=self.n_components,
                covariance_type="spherical",
                random_state=self.random_state,
            )

            gmm_refit.fit(df["values"].values.reshape(-1, 1))

        else:
            gmm_refit = GaussianMixture(
                n_components=self.n_components,
                covariance_type="spherical",
                random_state=self.random_state,
            )

            gmm_refit.fit(series.values.reshape(-1, 1))

        return gmm_refit

    def fit(self, X_train):
        """
        Method to fit the ADGMM model on the train data.

        Args:

            X_train : a dictionary
                The dictionary keys should be bin numbers (1, 2, 3, ...).
                The dictionary values should be pandas Series containing the data corresponding
                to the bin numbers.

        Returns:

            None
        """

        self.model_dict = {}
        self.train_bins = set(X_train.keys())

        for k, v in X_train.items():
            self.model_dict["model" + str(k)] = self.build(v)

        return None

    def predict(self, X_test):
        """
        Method to predict outliers on the test data.

        Args:

            X_test : a dictionary
                The dictionary keys should be bin numbers (1, 2, 3, ...).
                The dictionary values should be pandas Series containing the data corresponding
                to the bin numbers.

        Returns:

            binwise_outliers : a dictionary
                The dictionary keys are the bin numbers (1, 2, 3, ...).
                The dictionary values are pandas Series containing detected outliers corresponding
                to the bin numbers.

            total_outliers : a pandas Series storing all the detected outliers.

            binwise_outliers_filtered : a dictionary
                The dictionary keys are the bin numbers (1, 2, 3, ...).
                The dictionary values are pandas Series containing data excluding outliers
                corresponding to the bin numbers.

            total_outliers_filtered : a pandas Series storing the data excluding outliers.
        """

        binwise_outliers = {}
        binwise_outliers_filtered = {}
        total_outliers = pd.Series()
        total_outliers_filtered = pd.Series()

        self.test_bins = set(X_test.keys())
        bin_intersect = list(self.train_bins & self.test_bins)
        bin_diff = list(self.train_bins - self.test_bins)

        if len(bin_diff):
            logging.warning(
                """Predictions could not be made for the following bin numbers as they were
                absent in the training set \n {}""".format(
                    bin_diff
                )
            )

        for bin_num in bin_intersect:
            score = self.model_dict["model" + str(bin_num)].score_samples(
                X_test[str(bin_num)].values.reshape(-1, 1)
            ) ** (2 * self.exponential_scale_factor)
            scaled_score = pd.Series(
                MinMaxScaler().fit_transform(score.reshape(-1, 1)).reshape((-1)) * 10,
                index=X_test[str(bin_num)].index,
            )
            total_outliers = total_outliers.append(
                X_test[str(bin_num)][scaled_score >= self.val_thresh]
            )
            total_outliers_filtered = total_outliers_filtered.append(
                X_test[str(bin_num)][scaled_score < self.val_thresh]
            )
            binwise_outliers[str(bin_num)] = X_test[str(bin_num)][
                scaled_score >= self.val_thresh
            ]
            binwise_outliers_filtered[str(bin_num)] = X_test[str(bin_num)][
                scaled_score < self.val_thresh
            ]

        return (
            binwise_outliers,
            total_outliers,
            binwise_outliers_filtered,
            total_outliers_filtered,
        )
