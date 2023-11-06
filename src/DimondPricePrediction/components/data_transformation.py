import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.DimondPricePrediction.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')




class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info("Data Transformation initiated")

            #define which columns shpuld be ordinally encoded and which should be scaled
            categorical_cols =['cut','color','clarity']
            numerical_cols =['carat','depth','table','x','y','z']

            #define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good','Very Good','Premium','Ideal']
            clarity_categories = ['I1', 'SI2',  'SI1','VS2' , 'VS1','VVS2', 'VVS1', 'IF']
            color_categories =['J', 'I','H','G','F', 'E', 'D'  ]

            logging.info("Pipeline initiated")

            #numerical pipeline
            num_pipeline = Pipeline(
                steps=[

                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            #categorical pipeline
            cat_pipeline = Pipeline(
                steps=[

                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )
                            
            preprocessor = ColumnTransformer(   
                [
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',cat_pipeline,categorical_cols)
                ]
            )
            return preprocessor
        

        except Exception as e:
            logging.info("Exception occured in the get_data_transformation")

            raise customexception(e,sys)



    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")
            logging.info(f"Train Dataframe Head :\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head :\n{test_df.head().to_string()}")

            preprocessing_obj=self.get_data_transformation()

            target_column_name='price'
            drop_columns=[target_column_name,'id']

            input_feature_train_df =train_df.drop(columns=drop_columns)
            input_feature_test_df =test_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]

            input_feature_train_arr =preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr =preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets")

            save_object(
                file_path =self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")

            raise customexception(e,sys)