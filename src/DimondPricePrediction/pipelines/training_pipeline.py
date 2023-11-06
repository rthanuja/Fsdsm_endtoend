from src.DimondPricePrediction.components.data_ingestion import DataIngestion
from src.DimondPricePrediction.components.data_transformation import DataTransformation
from src.DimondPricePrediction.components.model_trainer import ModelTrainer

import os
import sys
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
import pandas as pd

obj=DataIngestion()

train_data_path,test_data_path = obj.initiate_data_ingestion()

DataTransformation_obj=DataTransformation()

train_arr,test_arr = DataTransformation_obj.initialize_data_transformation(train_data_path,test_data_path)

ModelTrainer_obj =ModelTrainer()

ModelTrainer_obj.initiate_model_training(train_arr,test_arr)