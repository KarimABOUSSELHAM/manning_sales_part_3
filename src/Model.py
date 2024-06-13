# import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import optuna
import joblib
import os
import ast
import mlflow

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAPE, MASE

from utils.extract_config import configfile
from utils.compute_metrics import compute_wmae, compute_wmape

# suppress pandas SettingWithCopyWarning 
pd.options.mode.chained_assignment = None

# set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

# get configurations
configfile = configfile()
model_path = configfile.get('model', 'model_path')
prediction_length = int(configfile.get('model', 'prediction_length'))
context_length = int(configfile.get('model', 'context_length'))
snaive_length = int(configfile.get('model', 'snaive_length'))
training_folder = configfile.get('model', 'training_folder')
model_folder = configfile.get('model', 'model_folder')
seed = int(configfile.get('model', 'seed'))
learning_rate = ast.literal_eval(configfile.get('model','learning_rate'))
widths = ast.literal_eval(configfile.get('model','widths'))

optimization_trials = int(configfile.get('optimization', 'optimization_trials'))
num_blocks_param = ast.literal_eval(configfile.get('optimization','num_blocks'))
num_block_layers_param = ast.literal_eval(configfile.get('optimization','num_block_layers'))
expansion_coefficient_lengths_T_param = ast.literal_eval(configfile.get('optimization','expansion_coefficient_lengths_T'))
expansion_coefficient_lengths_S_param = ast.literal_eval(configfile.get('optimization','expansion_coefficient_lengths_S'))

mlflow_tracking_uri = configfile.get('optimization', 'mlflow_tracking_uri')
mlflow_training_experiment_name = configfile.get('optimization', 'mlflow_training_experiment_name')


class NBeatsModel:
    """
    Class to handle model training and prediction
    """
    def __init__(self, model_path=model_path):
        self.current_model_path = model_path
        self.model = NBeats.load_from_checkpoint(model_path)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.training_df = None
        self.validation_df = None
        self.test_df = None

    def predict(self, data_df):
        """
        Performs prediction 

        Parameters:
        data_df (pandas.DataFrame): Data to predict on

        Returns:
        predict_df (pandas.DataFrame): Dataframe with predictions
        """
        # check the min and max date
        max_time_idx = data_df.time_idx.max()
        min_max_date_df = data_df.groupby('series_id').agg({'time_idx': [np.min, np.max]}).reset_index()
        min_max_date_df.columns = ['series_id', 'min', 'max']
        min_max_date_df['diff'] = min_max_date_df['max'] - min_max_date_df['min']
        min_diff = min_max_date_df['diff'].min()
        future_df = data_df[data_df.time_idx >= (max_time_idx - prediction_length + 1)]
        training_df = data_df[data_df.time_idx < (max_time_idx - prediction_length + 1)]

        # if sufficient data, use N-BEATS model for prediction
        if min_diff >= (self.prediction_length + self.context_length):
            predict_df = self._return_nbeats_predictions(data_df, future_df)
            logger.info("predictions made using N-BEATS")

        # else if insufficient data for N-BEATS but enough for sNaive then use sNaive
        elif min_diff >= (self.prediction_length + snaive_length):
            predict_df = self._return_snaive_predictions(training_df, future_df)
            logger.info("predictions made using SNaive")

        # else raise error
        else:
            error = "insufficient data for prediction"
            logger.error(error)
            raise Exception(error)
        
        predict_df = predict_df.loc[:, ['store_id', 'cat_id', 'date', 'pred']]
        now = datetime.now()
        predict_df['creation_time'] = now.strftime("%m/%d/%Y, %H:%M:%S")
        
        return predict_df

    def _return_nbeats_predictions(self, test_dataloader, test_df):
        """
        Given the trained trainer, test_dataloader and dataframe, 
        return a dataframe containing the predicted N-BEATS values

        Parameters:
        test_dataloader (torch.utils.data.DataLoader): Data to predict on
        test_df (pandas.DataFrame): Dataframe with metadata for the prediction period

        Returns:
        nbeats_test_df (pandas.DataFrame): Dataframe with predictions
        """
        # extract predictions from model
        predictions, index = self.model.predict(test_dataloader, return_index=True)

        # merge predictions and actual data into single dataframe
        time_idx_start = index.loc[0, 'time_idx']
        time_idx_end = time_idx_start + len(predictions[0])
        predictions_df_wide = pd.DataFrame(predictions.numpy(), columns=range(time_idx_start, time_idx_end))
        predictions_df_wide['series_id'] = index['series_id']
        predictions_df = predictions_df_wide.melt(id_vars=['series_id'])
        predictions_df.rename(columns={'variable':'time_idx'}, inplace=True)
        nbeats_test_df = test_df.merge(predictions_df, on=['series_id', 'time_idx'])
        nbeats_test_df.rename(columns={'value': 'pred'}, inplace=True)
        return nbeats_test_df

    def _return_snaive_predictions(self, training_df, test_df):
        """
        Implement the sNaive method by returning corresponding weekday sales in the last 7 days.
        Assumes weekly seasonality.

        Parameters:
        training_df (pandas.DataFrame): Data to predict on
        test_df (pandas.DataFrame): Dataframe with metadata for the prediction period

        Returns:
        snaive_test_df (pandas.DataFrame): Dataframe with predictions
        """
        training_df['dayofweek'] = training_df['date'].dt.weekday
        series_list = training_df['series_id'].unique()
        snaive_pred_list = []
        for series in series_list:
            training_df_series = training_df.loc[training_df.series_id==series]
            training_df_series.sort_values(by='date', ascending=False, inplace=True)
            last_week_df = training_df_series[:7][['dayofweek', 'series_id','sales']]
            snaive_pred_list.append(last_week_df)
        snaive_pred_df = pd.concat(snaive_pred_list)
        snaive_pred_df.rename(columns={'sales':'pred'}, inplace=True)
        test_df['dayofweek'] = test_df['date'].dt.weekday
        snaive_test_df = test_df.merge(snaive_pred_df, on=['series_id', 'dayofweek'], how='left')
        return snaive_test_df

    def _nbeats_modeler(self, training_df, validation_df, test_df, 
                        max_encoder_length=context_length,
                        num_blocks=[1,1], num_block_layers=[3,3],
                        expansion_coefficient_lengths=[3,3], 
                        batch_size=256, max_epochs=5, loss=MASE()):
        """
        Setup the N-BEATS model, trainer and dataloaders given the training, validation 
        and test dataframes, and parameters

        Parameters:
        training_df (pandas.DataFrame): Data for training
        validation_df (pandas.DataFrame): Data for validation
        test_df (pandas.DataFrame): Data for testing
        max_encoder_length (int): Number of time units that condition the predictions
        num_blocks (list):  The number of blocks per stack
        num_block_layers (list): Number of fully connected layers with ReLu activation per block.
        expansion_coefficient_lengths (list): Degree of modeling flexibility
        batch_size (int): Number of training examples utilized in one iteration
        max_epochs (int): Maximum number of epochs for training
        loss : PyTorch Forecasting metrics class
        """
        context_length = max_encoder_length

        # calculate the time indexes that the validation and test data start from
        val_idx = validation_df['time_idx'].min()
        test_idx = test_df['time_idx'].min()

        # setup Pytorch Forecasting TimeSeriesDataSet for training data
        training_data = TimeSeriesDataSet(
            training_df,
            time_idx="time_idx",
            target="sales",
            group_ids=["series_id"],
            time_varying_unknown_reals=["sales"],
            max_encoder_length=context_length,
            max_prediction_length=self.prediction_length,
            target_normalizer=GroupNormalizer(groups=['series_id']),
        )

        # setup Pytorch Forecasting TimeSeriesDataSet for validation and test data
        validation_data = TimeSeriesDataSet.from_dataset(training_data, pd.concat([training_df, validation_df]), min_prediction_idx=val_idx)
        test_data = TimeSeriesDataSet.from_dataset(training_data, pd.concat([training_df, validation_df, test_df]), min_prediction_idx=test_idx)

        # convert data to dataloader
        self.train_dataloader = training_data.to_dataloader(train=True, batch_size=batch_size)
        self.val_dataloader = validation_data.to_dataloader(train=False, batch_size=batch_size)
        self.test_dataloader = test_data.to_dataloader(train=False, batch_size=batch_size)

        pl.seed_everything(seed)  # set seed

        checkpoint_callback = ModelCheckpoint(monitor="val_loss")  # Init ModelCheckpoint callback, monitoring 'val_loss'
        logger = TensorBoardLogger(training_folder)  # log to tensorboard

        # setup PyTorch Lightning Trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=torch.cuda.device_count(),
            gradient_clip_val=1,
            callbacks=[checkpoint_callback],
            logger=logger,
        )

        # setup NBEATS model using PyTorch Forecasting's N-Beats class
        self.model = NBeats.from_dataset(
            training_data,
            num_blocks=num_blocks,
            num_block_layers=num_block_layers,
            expansion_coefficient_lengths=expansion_coefficient_lengths,
            learning_rate=learning_rate,
            log_interval=-1,
            log_val_interval=1,
            widths=widths,
            loss=loss,
            logging_metrics=torch.nn.ModuleList([MAPE()]),
        )

    def train(self, training_df, validation_df, test_df, 
              num_blocks=[1,1], num_block_layers=[3,3], expansion_coefficient_lengths=[3,3]):
        """
        Performs training

        Parameters:
        training_df (pandas.DataFrame): Data for training
        validation_df (pandas.DataFrame): Data for validation
        test_df (pandas.DataFrame): Data for testing
        num_blocks (list):  The number of blocks per stack
        num_block_layers (list): Number of fully connected layers with ReLu activation per block.
        expansion_coefficient_lengths (list): Degree of modeling flexibility
        """
        self._nbeats_modeler(training_df, validation_df, test_df, 
                             num_blocks=num_blocks, num_block_layers=num_block_layers, expansion_coefficient_lengths=expansion_coefficient_lengths)
        self.trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)
        
        # get best model based on validation loss
        self.best_model_path = self.trainer.checkpoint_callback.best_model_path 
        self.model = NBeats.load_from_checkpoint(self.best_model_path)

    def hyperparameter_tuning(self, training_df, validation_df, test_df):
        """
        Performs hyperparameter tuning

        Parameters:
        training_df (pandas.DataFrame): Data for training
        validation_df (pandas.DataFrame): Data for validation
        test_df (pandas.DataFrame): Data for testing
        """
        self.training_df = training_df
        self.validation_df = validation_df
        self.test_df = test_df

        # set MLflow path to log data
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # # set experiment name to organize runs
        mlflow.set_experiment(mlflow_training_experiment_name)
        experiment = mlflow.get_experiment_by_name(mlflow_training_experiment_name)

        study = optuna.create_study(direction="minimize")
        # study = joblib.load("study.pkl")  # comment above and uncomment to load existing study

        # start optimizing with number of trials = optimization_trials
        n_trials = optimization_trials

        # MLflow auto logging feature for PyTorch 
        mlflow.pytorch.autolog(log_models=False)

        # start tuning
        step = 1
        for _ in range(step, max(n_trials+1, step), step):
            study.optimize(lambda trial: objective(trial, training_df, validation_df, test_df, experiment), n_trials=step)

            # save optuna study to have option of resuming tuning
            joblib.dump(study, os.path.join(model_folder, "study.pkl"))

        # evaluate current model on test data for comparison against best experiment
        with mlflow.start_run(experiment_id = experiment.experiment_id):
            self.model = NBeats.load_from_checkpoint(self.current_model_path)
            nbeats_test_df = self._return_nbeats_predictions(pd.concat([training_df, validation_df, test_df]), test_df)
            wmae = compute_wmae(test_df, nbeats_test_df, 'sales', 'pred')
            mlflow.log_metric('test_wmae', wmae)


def objective(trial, training_df, validation_df, test_df, experiment):
    """
    Objective function to be used by Optuna to perform hyperparameter tuning
    """
    """
    Objective function to be used by Optuna to perform hyperparameter tuning

    Parameters:
    trial : Optuna trial object to evaluate objective function
    training_df (pandas.DataFrame): Data for training
    validation_df (pandas.DataFrame): Data for validation
    test_df (pandas.DataFrame): Data for testing
    experiment : MLflow experiment object

    Returns:
    val_wmae (float): Weighted mean absolute error calculated on validation data
    """
    # define hyperparamter space
    params = {
        "num_blocks": trial.suggest_int("num_blocks", num_blocks_param['start'], num_blocks_param['end'], step=num_blocks_param['step']),
        "num_block_layers": trial.suggest_int("num_block_layers", num_block_layers_param['start'], num_block_layers_param['end'], step=num_block_layers_param['step']),
        "expansion_coefficient_lengths_T": trial.suggest_int("expansion_coefficient_lengths_T", 
                                                             expansion_coefficient_lengths_T_param['start'], 
                                                             expansion_coefficient_lengths_T_param['end'], 
                                                             step=expansion_coefficient_lengths_T_param['step']),
        "expansion_coefficient_lengths_S": trial.suggest_int("expansion_coefficient_lengths_S", 
                                                             expansion_coefficient_lengths_S_param['start'], 
                                                             expansion_coefficient_lengths_S_param['end'], 
                                                             step=expansion_coefficient_lengths_S_param['step']),
    }

    with mlflow.start_run(experiment_id = experiment.experiment_id):

        # train model with params selected by Optuna
        expansion_coefficient_lengths = [params['expansion_coefficient_lengths_T'], params['expansion_coefficient_lengths_S']]
        num_blocks = [params['num_blocks'], params['num_blocks']]
        num_block_layers = [params['num_block_layers'], params['num_block_layers']]

        mlflow.log_params(trial.params)

        nbeatsmodel = NBeatsModel()
        nbeatsmodel.train(training_df, validation_df, test_df, 
                          num_blocks=num_blocks, num_block_layers=num_block_layers, 
                          expansion_coefficient_lengths=expansion_coefficient_lengths)

        # extract predictions and compute weighted mean absolute error (wmae)
        nbeats_val_df = nbeatsmodel._return_nbeats_predictions(nbeatsmodel.val_dataloader, validation_df)
        nbeats_test_df = nbeatsmodel._return_nbeats_predictions(nbeatsmodel.test_dataloader, test_df)
        val_wmae = compute_wmae(training_df, nbeats_val_df, 'sales', 'pred')
        test_wmae = compute_wmae(training_df, nbeats_test_df, 'sales', 'pred')

        mlflow.log_metric('val_wmae', val_wmae)
        mlflow.log_metric('test_wmae', test_wmae)
        mlflow.log_artifact(nbeatsmodel.best_model_path)

    return val_wmae