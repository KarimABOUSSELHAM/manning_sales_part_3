"""Script to setup database for exercise

"""

# import necessary libraries
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import logging


# set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# set database parameters
database = manning_sales
user = postgres
password = Karim
host= localhost

db_url = f"postgresql://{user}:{password}@{host}/{database}"

# establishing the connection
conn = psycopg2.connect(database=database, user=user, password=password, host=host, port= '5432')
logger.info("Database connection established successfully........")

# creating a cursor object using the cursor() method
cursor = conn.cursor()

# dropping SALE_CLEAN and PREDICTION tables if already exist
cursor.execute("DROP TABLE IF EXISTS SALE_CLEAN, PREDICTION")

# creating table as per requirement
create_pred_sql ='''CREATE TABLE PREDICTION(
   STORE_ID TEXT NOT NULL,
   CAT_ID TEXT NOT NULL,
   DATE DATE NOT NULL,
   PRED FLOAT NOT NULL,
   CREATION_TIME TIMESTAMP NOT NULL,
   PRIMARY KEY(DATE, STORE_ID, CAT_ID, CREATION_TIME)
)'''

create_sale_sql ='''CREATE TABLE SALE_CLEAN(
   STORE_ID TEXT NOT NULL,
   CAT_ID TEXT NOT NULL,
   DATE DATE NOT NULL,
   SALES FLOAT NOT NULL,
   IN_TRAINING BOOLEAN, 
   PRIMARY KEY(DATE, STORE_ID, CAT_ID)
)'''

cursor.execute(create_pred_sql)
logger.info("Prediction table created successfully........")

cursor.execute(create_sale_sql)
logger.info("Sales table created successfully........")

conn.commit()
# closing the connection
conn.close()

engine = create_engine(db_url)

# write sales_cleaned data to database
sales_df = pd.read_csv('../data/sales_cleaned.csv', parse_dates=['date'])
sales_df.to_sql('sale_clean', engine, if_exists='append', index=False)
logger.info("Sales data written to table sale_clean successfully........")

# write predictions data to database
predictions_df = pd.read_csv('../data/predictions.csv', parse_dates=['date', 'creation_time'])
predictions_df.to_sql('prediction', engine, if_exists='append', index=False)
logger.info("Predictions data written to table prediction successfully........")