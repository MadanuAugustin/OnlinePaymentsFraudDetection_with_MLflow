

import sys
import mysql.connector
import pandas as pd
from src.CreditScoreClassification.entity.config_entity import DataIngestionConfig
from src.CreditScoreClassification.Exception.custom_exception import CustomException
from src.CreditScoreClassification.logger_file.logger_obj import logger






class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.config = config



    # the below method is responsible for fetching the data from the SQL database

    def download_file(self):
        logger.info(f'-------Entered into download_file method------------')
        try:
            data_base_config = {
            'user': 'root',
            'password': '#@augustin#@7',
            'host': 'localhost',
            'database': 'onlinefrauddetection'
            }

            connection = mysql.connector.connect(**data_base_config)

            query = "SELECT * FROM online_fraud_detection"

            logger.info('---Requesting the SQL database------')

            raw_data = pd.read_sql(query, connection)

            logger.info('------Successfully fetched data from the SQL database----------')

            logger.info(f'---------The shape of the data is {raw_data.shape}')

            raw_data.to_csv(self.config.local_data_file, index=False)

            connection.close()

        except Exception as e:
            raise CustomException(e, sys)