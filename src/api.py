"""Script to setup REST API to serve predictions

"""

# import necessary libraries
import datetime
import json
import time
import logging
import uvicorn
from fastapi import FastAPI
from typing import Optional

from DataProcessor import DataProcessor
from utils.extract_config import configfile

# set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)



