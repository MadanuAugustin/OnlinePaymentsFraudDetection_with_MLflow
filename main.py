

import sys
from src.CreditScoreClassification.logger_file.logger_obj import logger
from src.CreditScoreClassification.pipeline.stage_01_dataIngestion import DataIngestionTrainingPipeline
from src.CreditScoreClassification.Exception.custom_exception import CustomException





STAGE_NAME = 'Data Ingestion Stage'


try:
    logger.info(f'--------------------stage {STAGE_NAME} started --------------------------')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f'----------------------stage {STAGE_NAME} completed ---------------------')
except Exception as e:
    raise CustomException(e, sys)