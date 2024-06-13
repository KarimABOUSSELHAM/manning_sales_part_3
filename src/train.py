"""Script for retraining model with hyperparameter tuning

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

