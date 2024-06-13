# import necessary libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging

from utils.extract_config import configfile

# suppress pandas SettingWithCopyWarning 
pd.options.mode.chained_assignment = None

# set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# get configurations
configfile = configfile()
db_url = configfile.get('database', 'database_url')
prediction_length = int(configfile.get('model', 'prediction_length'))
context_length = int(configfile.get('model', 'context_length'))


class DataProcessor:
    """
    Responsible for processing data
    """
    def __init__(self):
        self.engine = create_engine(db_url)  # sqlalchemy engine for managing database connection

    def get_data_for_prediction(self):
        """
        Retrieve latest sales data for prediction

        Returns:
        prediction_df (pandas.DataFrame): Dataframe of predictions data
        """
        # calculate latest date to filter sales data from based on latest sales data and context_length config
        date_df = self.read_from_db("select max(date) as date from sale_clean")
        max_date = pd.to_datetime(date_df.at[0, 'date'], format="%Y-%m-%d")
        filter_date = max_date - pd.Timedelta(f"{context_length} days")
        logger.info(f"filter date extracted as {filter_date}")

        # query and process sales data
        sales_df = self.read_from_db(f"select * from sale_clean where date >= '{filter_date}'")
        sales_df['date'] = pd.to_datetime(sales_df['date'], format="%Y-%m-%d")
        sales_df['series_id'] = sales_df['store_id'] + "_" + sales_df['cat_id']
        sales_df['time_idx'] = (sales_df['date'] - sales_df['date'].min()).dt.days
        test_date = max_date - pd.Timedelta(f"{prediction_length - 1} days")
        logger.info(f"data extracted and processed from sale_clean table")

        # create dataframe of future dates that require prediction
        future_df = sales_df.loc[sales_df.date >= test_date]
        future_df['time_idx'] = future_df['time_idx'] + prediction_length
        future_df['date'] = future_df['date'] + pd.Timedelta(f"{prediction_length} days")
        future_df['sales'] = 0

        prediction_df = pd.concat([sales_df, future_df]).reset_index(drop=True)

        return prediction_df
    
    def read_from_db(self, sql_script, **kwargs):
        """
        Query data from database given a SQL script

        Parameters:
        sql_script (str): SQL script to query database

        Returns:
        df (pandas.DataFrame): Dataframe of query results
        """
        df=pd.read_sql(sql_script,self.engine,**kwargs)

        return df

    def write_to_db(self, df, table_name, if_exists='append', index=False):
        """
        Write data to database

        Parameters:
        df (pandas.DataFrame): Dataframe of data to write to database
        table_name (str): Name of table to write to
        if_exists (str): One of 'fail', 'replace' or 'append'. How to behave if the table already exists.
        index (bool): Whether to write dataframe index as a column
        """
        df.to_sql(table_name,self.engine,if_exists=if_exists,index=index)


    def get_data_for_training(self, from_date=None, split_num=0):
        """
        Prepare data for training

        Parameters:
        from_date (str): Date to filter sales data from
        split_num (int): Cross-validation fold number

        Returns:
        training_df (pandas.DataFrame): Dataframe of training data
        validation_df (pandas.DataFrame): Dataframe of validation data
        test_df (pandas.DataFrame): Dataframe of test data
        """
        if from_date is None:
            sales_df = self.read_from_db("select * from sale_clean")
        else:
            sales_df = self.read_from_db(f"select * from sale_clean where date >= '{from_date}'")
        sales_df['date'] = pd.to_datetime(sales_df['date'], format="%Y-%m-%d")
        sales_df['series_id'] = sales_df['store_id'] + "_" + sales_df['cat_id']
        sales_df['time_idx'] = (sales_df['date'] - sales_df['date'].min()).dt.days
        training_df, validation_df, test_df = self._get_cv_split(sales_df, split_num)
        return training_df, validation_df, test_df

    def get_data_for_eval(self):
        """
        Prepare data for evaluation

        Returns:
        evaluation_df (pandas.DataFrame): Dataframe of evaluation data
        training_df (pandas.DataFrame): Dataframe of training data
        """
        # calculate latest date to filter sales data from based on latest sales data and prediction_length config
        date_df = self.read_from_db("select max(date) as date from sale_clean")
        max_date = pd.to_datetime(date_df.at[0, 'date'], format="%Y-%m-%d")
        filter_date = max_date - pd.Timedelta(f"{prediction_length - 1} days")
        logger.info(f"filter date extracted as {filter_date}")

        eval_data_script = f"""select s.store_id, s.cat_id, s.date, s.sales, p.pred, p.creation_time 
        from sale_clean as s 
        left join prediction as p 
        on p.cat_id=s.cat_id and p.store_id=s.store_id and p.date=s.date 
        where s.date >= '{filter_date}' """

        # query and process evaluation data
        evaluation_df = self.read_from_db(eval_data_script)
        evaluation_df['date'] = pd.to_datetime(evaluation_df['date'], format="%Y-%m-%d")
        evaluation_df['series_id'] = evaluation_df['store_id'] + "_" + evaluation_df['cat_id']

        logger.info(f"data extracted and processed from sale_clean table")

        # check whether there is prediction for each sales
        if evaluation_df[['sales', 'pred']].isnull().values.any():
            raise Exception("insufficient prediction or actual sales data")

        # calculate latest date to filter training sales data from and prediction_length config
        training_date_df = self.read_from_db("select max(date) as date from sale_clean where in_training = TRUE")
        training_max_date = pd.to_datetime(training_date_df.at[0, 'date'], format="%Y-%m-%d")
        training_filter_date = training_max_date - pd.Timedelta(f"{max(prediction_length, 28) - 1} days")

        # query and process training data
        training_df = self.read_from_db(f"select * from sale_clean where in_training = TRUE and date >= '{training_filter_date}'")
        training_df['date'] = pd.to_datetime(training_df['date'], format="%Y-%m-%d")
        training_df['series_id'] = training_df['store_id'] + "_" + training_df['cat_id']

        return evaluation_df, training_df

    def _get_cv_split(self, df, split_num, prediction_length=prediction_length, validation=True):
        """
        Implement train-test split given a cv fold number and return training, val and test data

        Parameters:
        df (pandas.DataFrame): Dataframe of data to split
        split_num (int): Cross-validation fold number
        prediction_length (int): Number of days to predict sales
        validation (bool): Whether to split with validation data or not

        Returns:
        training_df (pandas.DataFrame): Dataframe of training data
        validation_df (pandas.DataFrame): Dataframe of validation data
        test_df (pandas.DataFrame): Dataframe of test data
        """
        if 'series_id' not in df.columns:
            df['series_id'] = df['store_id'] + '_' + df['cat_id']
        series_list = df['series_id'].unique()

        test_list = []
        validation_list = []
        training_list = []

        for series in series_list:
            df_series = df.loc[df.series_id==series]
            max_date = df_series.date.max()
            min_date = df_series.date.min()
            test_lower_date = max_date - pd.Timedelta(f"{prediction_length*((split_num+1)*2-1)} days")
            test_upper_date = max_date - pd.Timedelta(f"{prediction_length*(split_num*2)} days")
            val_lower_date = max_date - pd.Timedelta(f"{prediction_length*(split_num+1)*2} days")
            if min(test_lower_date, test_upper_date) < min_date:
                raise Exception("Insufficient data for splitting")

            df_series_test = df_series.loc[(df_series.date > test_lower_date) & (df_series.date <= test_upper_date)]
            if validation:
                df_series_val = df_series.loc[(df_series.date > val_lower_date) & (df_series.date <= test_lower_date)]
                df_series_train = df_series.loc[df_series.date <= val_lower_date]
            else:
                df_series_val = pd.DataFrame()
                df_series_train = df_series.loc[df_series.date <= test_lower_date]
            test_list.append(df_series_test)
            validation_list.append(df_series_val)
            training_list.append(df_series_train)

        test_df = pd.concat(test_list)
        validation_df = pd.concat(validation_list)
        training_df = pd.concat(training_list)
        return training_df, validation_df, test_df