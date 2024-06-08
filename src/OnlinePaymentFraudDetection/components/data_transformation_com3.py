import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.OnlinePaymentFraudDetection.entity.config_entity import DataTransformationConfig
from src.OnlinePaymentFraudDetection.logger_file.logger_obj import logger
from src.OnlinePaymentFraudDetection.Exception.custom_exception import CustomException






class DataTransformation:
    def __init__(self, config : DataTransformationConfig):
        
        self.config = config




    def data_split(self):
        try:

            logger.info(f'-----------Entered data_split method---------------')
        
            data = pd.read_csv(self.config.local_data_file)

            data = data[["type", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud"]]

            data['type'] = data['type'].map({'PAYMENT': 3, 'TRANSFER' : 4, 'CASH_OUT' : 1, 'DEBIT' : 2, 'CASH_IN':0})

            train, test = train_test_split(data, test_size=0.2, random_state=42)

            train.to_csv(os.path.join(self.config.train_path, 'train.csv'), index = False, header = True)
        
            test.to_csv(os.path.join(self.config.test_path, 'test.csv'), index = False, header = True)

            logger.info(f'----------------saved train test data in csv format------------------')

            logger.info(f'------------The shape of the train data is {train.shape}')

            logger.info(f'--------------The shape of the test data is {test.shape}')

            logger.info(f'------------completed data splitting----------------------')

        except Exception as e:
            raise CustomException(e, sys)