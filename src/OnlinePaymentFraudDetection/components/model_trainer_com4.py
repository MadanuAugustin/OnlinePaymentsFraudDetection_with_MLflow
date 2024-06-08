





import pandas as pd
import joblib
import os
from src.OnlinePaymentFraudDetection.entity.config_entity import ModelTrainerConfig
from sklearn.tree import DecisionTreeClassifier
from src.OnlinePaymentFraudDetection.logger_file.logger_obj import logger


class ModelTrainer:
    def __init__(self, config : ModelTrainerConfig):
        self.config = config


    def initiate_model_training(self):

        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis = 1)
        test_x = test_data.drop([self.config.target_column], axis = 1)
        

        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        decision_tree = DecisionTreeClassifier()

        decision_tree.fit(train_x, train_y)

        logger.info(f'---------------Training of DecisionTree completed-----------------')

        joblib.dump(decision_tree, os.path.join(self.config.root_dir, self.config.model_name))


