"""
A helper function for creating seasonal binned data.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataBinning:
    """
    Utility class for preparing binned data.
    """

    def get_binned_data(self, X, agg_freq=None, bin_by=None, norm_data=False):
        """
        Prepare binned data.

        Args:

            X : a pandas Series.
                The series should have datetime index.

            agg_freq : str
                The string specification is identical to the
                pandas.DataFrame.resample method.
                Please refer to its documentation for all the available options.
                Examples:
                1. To aggregate and sum by every hour the string should be: "1H"
                2. To aggregate and sum by every 5 seconds the string should be: "5S"

            bin_by : str
                A string representing how the data points should be binned.
                The data can be binned by second, minute, hour, day or month.
                The string can take one of the following values:
                    "second", "minute", "hour", "day", "month"

            norm_data : bool, default = False
                Whether to apply min-max scaling on the data or not.
                Set value to True if you want to apply scaling.
                Set value to False if you do not want to apply scaling and leave data as is.

        Returns:

            binned_data: a dictionary
                The dictionary keys will be bin numbers (1, 2, 3, ...).
                The dictionary values will be pandas Series containing the data corresponding to the bin numbers.
                The range/values of bin numbers correspond to the bin_by criteria selected.
                For example, if bin_by = days, then the bin numbers will range from 1 to 31.
                This is assuming that there is atleast one data point for each bin.
                If a data point is missing that bin will not be created. So if there is no data for the 31st day,
                bin numbers will only be from 1 to 30.

        """

        X_copy = X.copy()
        binned_data = {}

        if agg_freq is not None:
            X_copy = X_copy.resample(agg_freq).sum()

        if norm_data == True:
            X_copy = pd.Series(
                MinMaxScaler().fit_transform(X_copy.values.reshape(-1, 1)).reshape(-1),
                index=X_copy.index,
            )

        unique_keys = eval("X_copy.index." + bin_by + ".unique().to_list()")

        for keys in unique_keys:
            binned_data[str(keys)] = eval("X_copy[X_copy.index." + bin_by + " == keys]")

        return binned_data
