import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_preprocessor_object(self):
        '''
        This Method  Will Give Data Transformation Object 
        
        '''

        try:
            logging.info("Data Transformation Started")

            catigorical_features = ['Brand', 'Model', 'Color', 'Material']

            numerical_features = ['Type', 'Gender', 'Size']

            # Creating Numerical And Catigorical pipline

            num_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
            ]
        )

            cato_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder(handle_unknown="ignore",sparse=False)),
                ("scaler",StandardScaler(with_mean=False))
            ]
        )

            # Creat Preprocessor obj
            preprocessing = ColumnTransformer([
                ("num_pipline",num_pipline,numerical_features),
                ("cato_pipline",cato_pipline,catigorical_features)
            ])

            return preprocessing
            logging.info("Pipline Complicted")


        except Exception as e:
            logging.info("Error Occured In Data Transformation")
            raise CustomException(e, sys)


    def initited_data_transformation(self,train_path,test_path):
        '''
        This Function will Apply Preprocessor Object And
        Transform Data

        '''
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read Traning And Testing Data Complited")
            logging.info(f"Traning Data head: \n{train_data.head().to_string()}")
            logging.info(f"Testing Data head: \n{test_data.head().to_string()}")

            logging.info("Obtaning Preprocessor Object")

            preprocessor_obj = self.get_preprocessor_object()

            target_columns = "Price"
            drop_columns = [target_columns]

            ## Split indiependent and Dependent Features
            input_features_train_data = train_data.drop(drop_columns,axis=1)
            target_feature_train_data = train_data[target_columns]

            ## Split indiependent and Dependent Features
            input_features_test_data = test_data.drop(drop_columns,axis=1)
            target_feature_test_data = test_data[target_columns]

            ## Apply Preprocessor Object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_data)

            logging.info("apply Preprocessor Object")

            ## Convert In To Array To Fast Process
            train_array = np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_feature_test_arr,np.array(target_feature_test_data)]

            ## Save Object In Pickle File
            save_object(filepath = self.data_transformation_config.preprocessor_obj_file_path,
             obj= preprocessor_obj)

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
             )

        except Exception as e:
            logging.info("Error Occured In Data Transformation")
            raise CustomException(e, sys)
        