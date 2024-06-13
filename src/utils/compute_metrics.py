"""Helper functions to compute metrics

"""

# import necessary libraries
import pandas as pd
import numpy as np


def compute_mae(training_df, prediction_test_df, y, y_hat, series_id):
    """
    Given a time series ID, compute the MAE for that time series and return the last 28-day training sales

    Parameters:
    training_df (pandas.DataFrame): Dataframe of training data
    prediction_test_df (pandas.DataFrame): Dataframe of predictions and actual test data
    y (str): Column name of actual sales
    y_hat (str): Column name of predicted sales
    series_id (str): ID of relevant time series, e.g., TX_1_FOODS

    Returns:
    mae (float): Mean absolute error metric
    total_sales (float): Total sales of last 28-day training data
    """
    prediction_test_df_series = prediction_test_df.loc[prediction_test_df.series_id==series_id]
    training_df_series = training_df.loc[training_df.series_id==series_id]
    training_df_series.sort_values(by='date', ascending=False, inplace=True)
    prediction_test_df_series['abs_error'] = (prediction_test_df_series[y_hat] - prediction_test_df_series[y]).abs()
    mae = prediction_test_df_series['abs_error'].mean()
    total_sales = training_df_series[:28]['sales'].sum()
    return mae, total_sales


def compute_wmae(training_df, prediction_test_df, y, y_hat):
    """
    Given a training and prediction data, compute the weighted MAE

    Parameters:
    training_df (pandas.DataFrame): Dataframe of training data
    prediction_test_df (pandas.DataFrame): Dataframe of predictions and actual test data
    y (str): Column name of actual sales
    y_hat (str): Column name of predicted sales

    Returns:
    wmae (float): Weighted mean absolute error metric
    """
    series_list = prediction_test_df.series_id.unique()
    sales_list = []
    mae_list = []
    for series in series_list:
        mae_series, total_sales_series = compute_mae(training_df, prediction_test_df, y, y_hat, series)
        mae_list.append(mae_series)
        sales_list.append(total_sales_series)
    overall_sales = np.sum(sales_list)
    weights_list = [s/overall_sales for s in sales_list]
    wmae_list = [a*b for a,b in zip(mae_list, weights_list)]
    wmae = np.sum(wmae_list)
    return wmae


def compute_mape(training_df, prediction_test_df, y, y_hat, series_id):
    """
    Given a time series ID, compute the MAPE for that time series and return the last 28-day training sales

    Parameters:
    training_df (pandas.DataFrame): Dataframe of training data
    prediction_test_df (pandas.DataFrame): Dataframe of predictions and actual test data
    y (str): Column name of actual sales
    y_hat (str): Column name of predicted sales
    series_id (str): ID of relevant time series, e.g., TX_1_FOODS

    Returns:
    mape (float): Mean absolute persentage error metric
    total_sales (float): Total sales of last 28-day training data
    """
    training_df_series = training_df.loc[training_df.series_id==series_id]
    training_df_series.sort_values(by='date', ascending=False, inplace=True)
    prediction_test_df_series = prediction_test_df.loc[prediction_test_df.series_id==series_id]
    prediction_test_df_series['abs_pct_error'] = ((prediction_test_df_series[y] - prediction_test_df_series[y_hat])/prediction_test_df_series[y]).abs()
    mape = prediction_test_df_series['abs_pct_error'].mean()
    total_sales = training_df_series[:28]['sales'].sum()
    return mape, total_sales


def compute_wmape(training_df, prediction_test_df, y, y_hat):
    """
    Given a training and prediction data, compute the weighted MAPE

    Parameters:
    training_df (pandas.DataFrame): Dataframe of training data
    prediction_test_df (pandas.DataFrame): Dataframe of predictions and actual test data
    y (str): Column name of actual sales
    y_hat (str): Column name of predicted sales

    Returns:
    wmape (float): Weighted mean absolute percentage error metric
    """
    series_list = prediction_test_df.series_id.unique()
    sales_list = []
    mape_list = []
    for series in series_list:
        mape_series, total_sales_series = compute_mape(training_df, prediction_test_df, y, y_hat, series)
        mape_list.append(mape_series)
        sales_list.append(total_sales_series)
    overall_sales = np.sum(sales_list)
    weights_list = [s/overall_sales for s in sales_list]
    wmape_list = [a*b for a,b in zip(mape_list, weights_list)]
    wmape = np.sum(wmape_list)
    return wmape


def compute_eval_data_ratio(training_df, evaluation_df):
    """
    Given a training and evaluation data, compute the data metrics

    Parameters:
    training_df (pandas.DataFrame): Dataframe of training data
    evaluation_df (pandas.DataFrame): Dataframe of evaluation data

    Returns:
    mean_sales_ratio (float): Ratio of Mean sales of the eval data to Mean sales of the latest 28 days of training data
    stdev_sales_ratio (float): Std dev of the sales for eval data to Std dev of the sales for the latest 28 days of training data
    """
    
    mean_sales_ratio = None
    stdev_sales_ratio = None

    return mean_sales_ratio, stdev_sales_ratio