import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,IsolationForest
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from src.utils import save_object,model_evaluation

@dataclass
class ModelTraningConfig:
    traning_model_file_obj = os.path.join("artifcats","model.pkl")

class ModelTraning:

    def __init__(self):
        self.model_traning_config = ModelTraningConfig()

    def start_model_traning(self,train_array,test_array):
        '''
        This Function Will Train The Multiple Model And Give Best Model

        '''
        try:
            logging.info("Split Dependent And Indipendent Features")
            X_train,y_train,X_test,y_test = (
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1],
            )

            models = {
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "Elastic_Net":ElasticNet(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
            }
            params = {
                "LinearRegression":{
        
                 },
                 "Lasso": {
                    "alpha": [0.01, 0.1, 1, 10]
                },
                "Ridge": {
                    "alpha": [0.01, 0.1, 1, 10]
                },
                "Elastic_Net": {
                    "alpha": [0.01, 0.1, 1, 10],
                    "l1_ratio": [0.2, 0.4, 0.6, 0.8]
                },
                "DecisionTreeRegressor": {
                    "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "splitter":['best','random'],
                    "max_depth": [8,15,20],
                    "min_samples_split": [2,3,4],
                    "min_samples_leaf": [2,4],
                    "max_features":["auto","sqrt","log2"]
                },
                "RandomForestRegressor":{
                    'n_estimators': [100, 200,250],
                    "criterion": ["squared_error", "friedman_mse"],
                    'max_depth': [8,15,20],
                    'min_samples_split': [2,3,4],
                    'min_samples_leaf': [2,4],
                },
                "GradientBoostingRegressor":{
                    'n_estimators': [100, 200,250],
                    'learning_rate':[0.1,0.01,0.001],
                    'loss':["squared_error", "absolute_error", "huber", "quantile"],
                    "criterion": ["squared_error", "friedman_mse"]
                },
                "KNeighborsRegressor":{
                    'n_neighbors': [5,6,8],
                    'weights':["uniform", "distance"],
                    'algorithm':["auto", "ball_tree", "kd_tree", "brute"],    
                }
            }

            model_report:dict = model_evaluation(X_train, y_train, X_test, y_test, models, params)

            ## To Get The Best Model Score Dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Name Is : {best_model_name},Accuracy_score: {best_model_score}")
            print("*"*100)
            logging.info(f"Best Model Found, Name Is : {best_model_name},Accuracy_score: {best_model_score}")

            save_object(filepath = self.model_traning_config.traning_model_file_obj,
             obj = best_model
             )

        except Exception as e:
            logging.info("Error Occured In Model Traning Stage")
            raise CustomException(e, sys)

