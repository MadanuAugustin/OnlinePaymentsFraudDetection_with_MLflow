




import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.OnlinePaymentFraudDetection.logger_file.logger_obj import logger
from src.OnlinePaymentFraudDetection.Exception.custom_exception import CustomException


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts//model_trainer//model.joblib'))
        self.preprocessorObj = joblib.load(Path('artifacts//data_transformation//preprocessor_obj.joblib'))


    # the below method takes the data from the user to predict

    def predictDatapoint(self, data):
        
        try:

            data_df = data.rename(columns = {0 : 'amount', 1 : 'oldbalanceOrg', 2 : 'newbalanceOrig',
                                             3 : 'type'})
            
            print(data_df)

            user_numeric_cols = data_df.drop(columns = ['type'], axis = 1)

            user_categoric_cols = data_df[['type']]

            transformed_numeric_cols = self.preprocessorObj.transform(user_numeric_cols)

            transformed_categoric_cols = user_categoric_cols['type'].map({'PAYMENT': 3, 'TRANSFER' : 4, 'CASH_OUT' : 1, 'DEBIT' : 2, 'CASH_IN':0})

            transformed_user_input = pd.DataFrame(np.c_[transformed_numeric_cols, transformed_categoric_cols])

            logger.info(f'---------Below is the transformed user input----------------')

            print(transformed_user_input)


            prediction = self.model.predict(transformed_user_input)

            list_output  = []

            if prediction == [0.]:
                list_output.append('Not_Fraud')
            elif prediction == [1.]:
                list_output.append('Fraud')

            logger.info(f'-----------Below output is predicted by the model---------------')

            print(list_output)

            return list_output
        
        
        except Exception as e:
            raise CustomException(e, sys)

