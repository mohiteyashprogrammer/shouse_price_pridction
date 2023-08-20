import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException

from src.utils import lode_object

class PredictPipline:

    def __init__(self):
        pass

    @staticmethod
    def predict(features):
        '''
        This function Will Predict The Output 
        Based On Input Features

        '''
        try:
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            preprocessor = lode_object(preprocessor_path)
            model = lode_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occured In Predict Pipline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
            Brand:str,
            Model:str,
            Type:int,
            Gender:int,
            Size:float,
            Color:str,
            Material:str
            ):

        self.Brand = Brand
        self.Model = Model
        self.Type = Type
        self.Gender = Gender
        self.Size = Size
        self.Color = Color
        self.Material = Material

    def get_data_as_data_frame(self):
        '''

        This Function Will Create Data Frame

        '''
        try:
            custom_data_input_dict = {
                "Brand":[self.Brand],
                "Model":[self.Model],
                "Type":[self.Type],
                "Gender":[self.Gender],
                "Size":[self.Size],
                "Color":[self.Color],
                "Material":[self.Material],
            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("Data Frame Gathered")
            return data

        except Exception as e:
            raise CustomException(e, sys)

        




