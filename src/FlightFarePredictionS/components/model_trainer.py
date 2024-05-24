
import pandas as pd
import numpy as np
import os
import sys
from src.FlightFarePredictionS.logger import logging
from src.FlightFarePredictionS.exception import customexception
from dataclasses import dataclass
from src.FlightFarePredictionS.utils.utils import save_object
from src.FlightFarePredictionS.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,X_train, y_train, X_test, y_test):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
             

            models={
            'Linear_regression':LinearRegression(), 
         'Elastic_net': ElasticNet(), 
         'Lasso':Lasso(), 
         'ridge':Ridge(), 
         'random_forest':RandomForestRegressor(), 
         'Gradient_boost': GradientBoostingRegressor(), 
         'Ada_boost': AdaBoostRegressor(), 
         'Decision_tree': DecisionTreeRegressor()
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)

        
    