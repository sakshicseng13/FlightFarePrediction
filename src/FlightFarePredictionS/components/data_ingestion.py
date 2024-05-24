
import pandas as pd
import numpy as np
from src.FlightFarePredictionS.logger import logging
from src.FlightFarePredictionS.exception import customexception
from dataclasses import dataclass
from pathlib import Path

import sys
import os


@dataclass
class DataIngestionconfig:
    data_path: str = os.path.join('notebooks/data', 'Data_Train.xlsx')
    data:str = os.path.join('artifact','rawData_Train.xlsx')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df = pd.read_excel(self.ingestion_config.data_path)
            logging.info('Dataset read as a pandas DataFrame')

            df.to_excel(self.ingestion_config.data,index=False,header=True)
            logging.info("Dataset stored in artifact folder as rawData_Train.xlsx")
            logging.info('Data ingestion is successful')
            return self.ingestion_config.data

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion Stage')
            raise CustomException(e, sys)
    