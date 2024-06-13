"""Script for making predictions

"""

# import necessary libraries
import traceback
import logging
from logging.handlers import TimedRotatingFileHandler

from DataProcessor import DataProcessor
from Model import NBeatsModel
from utils.extract_config import configfile


# get configurations
configfile = configfile()
log_path = configfile.get('logging', 'log_path')

# set logging
logFormatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

fileHandler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=90)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

logger.setLevel(logging.INFO)
logger.propagate = False

try:
    #read in the sales data
    data_preprocessor=DataProcessor()
    data_for_prediction=data_preprocessor.get_data_for_prediction()
    logger.info("Data for prediction has been red")

    #pass the data to the model 
    nbeats_model=NBeatsModel()
    df_pred=nbeats_model.predict(data_for_prediction)
    logger.info("Predictions have been obtained")

    #write the model predictions to the database
    data_preprocessor.write_to_db(df_pred,'predictions')
    logger.info("Predictions logged to the database")

except Exception:
    logger.error(traceback.format_exc())