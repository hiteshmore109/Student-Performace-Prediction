import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.utils import save_object
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformer(self):
        try:
            numerical_features = ["reading_score", "writing_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_transform = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            logging.info("Pipeline for numerical tranformation created")

            cat_transform = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Pipeline for categorical tranformation created")

            preprocessor = ColumnTransformer(
                [
                    ("num_transform", num_transform, numerical_features),
                    ("cat_transform", cat_transform, categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def transform_data(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read Train and Test data")

            preprocessor = self.data_transformer()
            logging.info("Received data transformer")

            target = "math_score"
            train_df_features = train_df.drop(columns=target)
            test_df_features = test_df.drop(columns=target)
            train_df_target = train_df[target]
            test_df_target = test_df[target]
            logging.info("Extracted features and target from train and test datasets")

            train_df_features_transformed = preprocessor.fit_transform(
                train_df_features
            )
            test_df_features_transformed = preprocessor.fit_transform(test_df_features)
            logging.info("Transformed the features of test and train data")

            train_transformed_arr = np.c_[
                train_df_features_transformed, np.array(train_df_target)
            ]
            test_transformed_arr = np.c_[
                test_df_features_transformed, np.array(test_df_target)
            ]
            logging.info("Preprocessing Completed")

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                object=preprocessor,
            )
            logging.info("Preprocessor saved successfully.")

            return (
                train_transformed_arr,
                test_transformed_arr,
                self.data_transformation_config.preprocessor_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
