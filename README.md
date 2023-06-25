# Gaussian Mixture Model for Anomaly Detection in Seasonal Time Series Data

## About
This repository is a Python implementation for the algorithm described in the paper [Using Gaussian Mixture Models to Detect Outliers in Seasonal Univariate Network Traffic](https://ieeexplore.ieee.org/document/8227312).

This is applicable for time series data with events exhibiting seasonal behaviour.

## Brief explanation
When we have time series data where the event of our interest shows seasonality, the method used in this paper considers building separate models for each season rather than a single model by considering the entire data.

Let's say we have time series data with a minute-level granularity for an event (no. of transactions, network traffic, etc.). If the usual behavior of this event changes every hour, then we build 24 models (since there are 24 hours in a day). The dataset for each hour is prepared by aggregating the data hourly. This is called data binning.

There are two main steps:
1. GMMs are built for training data in each time bin of seasonal time series data. Outliers or anomalies are detected and removed in this training data set by examining the probability associated with each data point.
2. GMMs are rebuilt after outliers are removed in historical or training data and the re-computed GMMs are used to detect outliers in test data.

## Code usage
If you don't have the data bins created, you may use the helper function `get_binned_data` in the code `utils/data_binning.py` to prepare the data for modelling.

Once you have the train data and the test data, instantiate the ADGMM class object in the code `src/gmm_ad.py`, and call the `fit` method to build the model. Next use the `predict` method to make the predictions. This is similar to how we use scikit-learn library.

Please refer to the funciton docstrings for more information on its parameters and usage.

## Tools and libraries used

Purpose | Tools/Libraries
---|---
Data manipulation | Pandas
Numerical computations | NumPy
Model building | Scikit-Learn






