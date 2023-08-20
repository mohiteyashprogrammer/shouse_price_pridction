import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import r2_score


# Function For Saving Pickle file
def save_object(filepath,obj):
    '''
    This Function  Save Pickle File

    '''
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def model_evaluation(X_train,y_train,X_test,y_test,models,params):
    '''
    This Function Will Train The model And Evaluate The Model
    Give R2 Score As Accuracy

    '''
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            ## Model Traning
            gs = GridSearchCV(model, para,cv=5)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            ## Make Prediction
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def lode_object(file_path):
    '''
    This  Function Will Loab Pickel File And 
    Read In binery Mode
    
    '''
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)