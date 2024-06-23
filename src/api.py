"""Script to setup REST API to serve predictions

"""

# import necessary libraries
import datetime
from json import loads
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

# get configurations
configfile = configfile()
db_url = configfile.get('database', 'database_url')

app=FastAPI()

#The followinf view function is just for test as per the project questions
@app.get('/')
def root():
    return {'message': 'hello world'}

@app.get('/predictions')
def get_predictions(start: Optional[datetime.date]= None, end: Optional[datetime.date]= None,
                    store_id: Optional[str]= None, cat_id: Optional[str]= None):
    """
    Retrieve predictions data from database based on query parameters

    Parameters:
    start (datetime.date): Starting date to query from. Optional.
    end (datetime.date): Ending date to query to. Optional.
    category (str): Category to filter. Optional.
    store (str): Store to filter. Optional.

    Returns:
    parsed (JSON): Predictions data and relevant schema 
    """
    start_time = time.time()
    dataprocessor=DataProcessor()

    # logic to read relevant predictions data based on query parameters
    if start is not None and end is not None:
        df_pred=dataprocessor.read_from_db(sql_script=f"distinct on (cat_id, store_id, date)* from prediction where date between {start} and {end} order by cat_id, store_id, date, creation_time desc", parse_dates=['date','creation_time'])
    elif start is None and end is not None:
        df_pred=dataprocessor.read_from_db(sql_script=f"distinct on (cat_id, store_id, date)* from prediction where date<= {end} order by cat_id, store_id, date, creation_time desc", parse_dates=['date','creation_time'])
    elif start is not None and end is None:
        df_pred=dataprocessor.read_from_db(sql_script=f"distinct on (cat_id, store_id, date)* from prediction where date>= {start} order by cat_id, store_id, date, creation_time desc", parse_dates=['date','creation_time'])
    else:
        df_pred=dataprocessor.read_from_db(sql_script="distinct on (cat_id, store_id, date)* from prediction order by cat_id, store_id, date, creation_time desc limit 100", parse_dates=['date','creation_time'])
    
    if cat_id is not None:
        df_pred=df_pred.loc[df_pred["cat_id"]==cat_id]
    if store_id is not None:
        df_pred=df_pred.loc[df_pred["store_id"]==store_id]
    # convert dataframe to JSON to serve data
    result=df_pred.to_json(orient="table",index=False)
    parsed=loads(result)

    logging.info(f"time taken: {time.time()-start_time}")
    return parsed


if __name__=='__main__':
    uvicorn.run("api:app", host="127.0.0.1", port=8000, log_level="info")



